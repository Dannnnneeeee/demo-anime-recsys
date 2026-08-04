"""
backend.py

Mesin rekomendasi untuk deployment. Logika di sini adalah replika langsung
dari notebook_08_inference_online: confidence, fold-in ALS, skor konten,
penyusunan fitur, reranking CatBoost, dan penjelasan SHAP.

Modul ini tidak bergantung pada Streamlit sehingga dapat diuji secara
terpisah lewat baris perintah biasa sebelum dipakai di aplikasi web.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from catboost import CatBoostRanker, Pool


class RecommenderEngine:
    def __init__(self, bundle_dir):
        self.dir = Path(bundle_dir)
        self._load()

    def _load(self):
        d = self.dir
        with open(d / "config.json") as f:
            self.config = json.load(f)

        self.alpha = self.config["alpha"]
        self.reg = self.config["reg"]
        self.n_candidates = self.config["n_candidates"]
        self.top_k = self.config["top_k"]
        self.max_liked = self.config["max_liked"]
        self.default_seed_score = self.config["default_seed_score"]
        self.feature_order = self.config["feature_order"]

        self.item_factors = np.load(d / "item_factors.npy")
        self.n_items = self.item_factors.shape[0]

        user_factors_demo = np.load(d / "user_factors_demo.npy")
        user_ids_demo = np.load(d / "user_ids_demo.npy")
        self.user_factors = {
            int(uid): user_factors_demo[i] for i, uid in enumerate(user_ids_demo)
        }
        self.demo_user_ids = sorted(self.user_factors.keys())

        self.content_matrix = sp.load_npz(d / "item_content_matrix.npz").tocsr()

        self.item_features = pl.read_parquet(d / "item_features.parquet").sort("item_idx")
        titles = pl.read_parquet(d / "titles.parquet")
        self.title_by_item = dict(zip(titles["item_idx"].to_list(), titles["title"].to_list()))
        self.adult_by_item = dict(zip(titles["item_idx"].to_list(), titles["is_adult"].to_list()))

        genres_df = pl.read_parquet(d / "item_genres.parquet")
        self.genres_by_item = {
            row["item_idx"]: (row["genres"] or [])
            for row in genres_df.iter_rows(named=True)
        }

        history = pl.read_parquet(d / "user_history_demo.parquet")
        self.train_items_by_user = {
            row["user_idx"]: set(row["train_items"]) for row in history.iter_rows(named=True)
        }
        self.liked_items_by_user = {
            row["user_idx"]: list(row["liked_items"]) for row in history.iter_rows(named=True)
        }

        test_rel = pl.read_parquet(d / "test_relevance_demo.parquet")
        self.relevance_by_user = {}
        for row in test_rel.iter_rows(named=True):
            u = row["user_idx"]
            grade = self._grade_of(row["score"])
            self.relevance_by_user.setdefault(u, {})[row["item_idx"]] = grade

        f = self.item_factors
        self.YtY = f.T @ f
        self.lamI = self.reg * np.eye(f.shape[1])

        titles_display = titles.filter(pl.col("title").is_not_null())
        self.browsable_items = list(zip(
            titles_display["item_idx"].to_list(),
            titles_display["title"].to_list(),
        ))

    @staticmethod
    def _grade_of(score):
        if score >= 9:
            return 3
        if score >= 7:
            return 2
        if score >= 5:
            return 1
        return 0

    def als_scores_for_user(self, user_idx):
        vec = self.user_factors[user_idx]
        return self.item_factors @ vec

    def fold_in(self, seed_items, seed_scores):
        resolved = np.array(
            [s if s is not None else self.default_seed_score for s in seed_scores],
            dtype=np.float64,
        )
        conf = 1.0 + self.alpha * resolved
        Ys = self.item_factors[seed_items]
        A = self.YtY + (Ys.T * (conf - 1.0)) @ Ys + self.lamI
        b = (Ys.T * conf) @ np.ones(len(seed_items))
        x_u = np.linalg.solve(A, b)
        return self.item_factors @ x_u

    def content_score(self, liked_items, candidates):
        liked = list(liked_items)[: self.max_liked]
        if not liked:
            return np.zeros(len(candidates))
        sim = self.content_matrix[list(candidates)] @ self.content_matrix[liked].T
        return np.asarray(sim.todense()).max(axis=1)

    @staticmethod
    def top_n_excluding(scores, exclude, n):
        order = np.argsort(-scores)
        out = []
        for idx in order:
            if int(idx) not in exclude:
                out.append(int(idx))
            if len(out) == n:
                break
        return np.array(out)

    def _assemble(self, candidates, als_scores_for_cand, content_scores):
        cand_df = pl.DataFrame({"item_idx": list(candidates)})
        joined = cand_df.join(self.item_features, on="item_idx", how="left")
        joined = joined.with_columns(
            pl.Series("als", als_scores_for_cand),
            pl.Series("content", content_scores),
        )
        return joined.select(self.feature_order + ["item_idx"])

    def _rerank_scores(self, feat_df):
        pool = Pool(feat_df.select(self.feature_order).to_pandas(), cat_features=["type"])
        return self.model.predict(pool)

    def _shap_contributions(self, feat_df):
        pool = Pool(feat_df.select(self.feature_order).to_pandas(), cat_features=["type"])
        shap = self.model.get_feature_importance(pool, type="ShapValues")
        base = shap[:, -1]
        contribs = shap[:, :-1]
        return contribs, base

    def load_model(self):
        self.model = CatBoostRanker()
        self.model.load_model(str(self.dir / "reranker.cbm"))
        return self

    def _driver_label(self, contrib_row):
        idx = int(np.argmax(np.abs(contrib_row)))
        name = self.feature_order[idx]
        labels = {
            "als": "kolaboratif",
            "content": "konten",
            "log_pop": "populer",
            "bayes_wr": "rating tinggi",
            "wilson": "rating konsisten",
            "mean_score": "skor tinggi",
            "year": "era rilis",
            "type": "tipe",
            "is_adult": None,
        }
        return labels.get(name, name)

    def _finish(self, candidates, rerank_scores, feat_df, als_full, content_scores, hide_adult):
        order = np.argsort(-rerank_scores)[: self.top_k * 2]
        contribs, base = self._shap_contributions(feat_df)
        item_idx_list = feat_df["item_idx"].to_list()

        rows = []
        for i in order:
            item_idx = int(candidates[i])
            if hide_adult and self.adult_by_item.get(item_idx, False):
                continue

            driver = self._driver_label(contribs[i])
            feat_row = feat_df.row(i, named=True)
            base_val = float(base[i])
            feature_details = [
                {
                    "fitur": name,
                    "nilai": feat_row[name],
                    "kontribusi": round(float(contribs[i][j]), 6),
                }
                for j, name in enumerate(self.feature_order)
            ]
            feature_details.sort(key=lambda x: abs(x["kontribusi"]), reverse=True)
            total_rekonstruksi = base_val + sum(f["kontribusi"] for f in feature_details)

            rows.append({
                "peringkat": len(rows) + 1,
                "item_idx": item_idx,
                "judul": self.title_by_item.get(item_idx, "(tanpa judul)"),
                "skor_als": round(float(als_full[item_idx]), 4),
                "skor_konten": round(float(content_scores[i]), 4),
                "skor_akhir": round(float(rerank_scores[i]), 4),
                "pendorong": driver,
                "shap": {
                    "base_value": round(base_val, 6),
                    "fitur": feature_details,
                    "total_rekonstruksi": round(total_rekonstruksi, 6),
                },
            })
            if len(rows) == self.top_k:
                break

        top_items = [r["item_idx"] for r in rows]
        return rows, top_items

    def ndcg_personal(self, user_idx, ranked_items):
        rel = self.relevance_by_user.get(user_idx)
        if not rel:
            return None
        dcg = sum(
            rel.get(item, 0) / np.log2(pos + 2)
            for pos, item in enumerate(ranked_items[: self.top_k])
        )
        ideal_grades = sorted(rel.values(), reverse=True)[: self.top_k]
        idcg = sum(g / np.log2(pos + 2) for pos, g in enumerate(ideal_grades))
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def recommend_existing_user(self, user_idx, hide_adult=True):
        import time
        t0 = time.perf_counter()

        als_full = self.als_scores_for_user(user_idx)
        seen = self.train_items_by_user.get(user_idx, set())
        candidates = self.top_n_excluding(als_full, seen, self.n_candidates)

        liked = self.liked_items_by_user.get(user_idx, [])
        content_scores = self.content_score(liked, candidates)

        feat_df = self._assemble(candidates, als_full[candidates], content_scores)
        rerank_scores = self._rerank_scores(feat_df)

        rows, top_items = self._finish(candidates, rerank_scores, feat_df, als_full, content_scores, hide_adult)
        ndcg = self.ndcg_personal(user_idx, top_items)

        latency_ms = (time.perf_counter() - t0) * 1000
        return rows, ndcg, latency_ms

    def recommend_guest(self, seed_titles_scores, hide_adult=True):
        import time
        t0 = time.perf_counter()

        seed_items = [item for item, _ in seed_titles_scores]
        seed_scores = [score for _, score in seed_titles_scores]

        als_full = self.fold_in(np.array(seed_items), seed_scores)
        seen = set(seed_items)
        candidates = self.top_n_excluding(als_full, seen, self.n_candidates)

        content_scores = self.content_score(seed_items, candidates)

        feat_df = self._assemble(candidates, als_full[candidates], content_scores)
        rerank_scores = self._rerank_scores(feat_df)

        rows, _ = self._finish(candidates, rerank_scores, feat_df, als_full, content_scores, hide_adult)

        latency_ms = (time.perf_counter() - t0) * 1000
        return rows, latency_ms

    def search_titles(self, query, limit=20):
        query = query.strip().lower()
        if not query:
            return self.browsable_items[:limit]
        matches = [
            (item_idx, title)
            for item_idx, title in self.browsable_items
            if query in title.lower()
        ]
        return matches[:limit]

    def get_user_history(self, user_idx, limit=20):
        liked = self.liked_items_by_user.get(user_idx, [])
        all_seen = self.train_items_by_user.get(user_idx, set())
        titles = [
            {"item_idx": it, "judul": self.title_by_item.get(it, "(tanpa judul)")}
            for it in liked[:limit]
        ]
        return {
            "total_interaksi": len(all_seen),
            "total_disukai": len(liked),
            "contoh_disukai": titles,
        }

    def genre_profile(self, item_ids, top_n=8):
        item_ids = list(item_ids)
        if not item_ids:
            return []
        counter = {}
        for it in item_ids:
            for g in self.genres_by_item.get(it, []):
                counter[g] = counter.get(g, 0) + 1
        total = len(item_ids)
        profile = [
            {"genre": g, "jumlah": c, "persentase": round(100 * c / total, 1)}
            for g, c in counter.items()
        ]
        profile.sort(key=lambda x: x["jumlah"], reverse=True)
        return profile[:top_n]