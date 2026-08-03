"""
prepare_artifacts.py

Membuat paket artefak berukuran kecil untuk deployment web, diambil dari
artefak lengkap hasil penelitian. Dijalankan sekali secara lokal.

Isi paket:
  - user_factors_demo.npy      : vektor laten 1000 pengguna terpilih
  - user_ids_demo.npy          : user_idx asli dari 1000 pengguna tersebut
  - item_factors.npy           : vektor laten seluruh item (utuh)
  - item_content_matrix.npz    : matriks konten seluruh item (utuh)
  - item_features.parquet      : fitur item untuk reranker (log_pop, wilson, dst)
  - titles.parquet             : item_idx, judul, is_adult
  - test_relevance_demo.parquet: ground truth test_A, hanya untuk 1000 pengguna terpilih
  - reranker.cbm               : model reranker (disalin apa adanya)
  - config.json                : parameter (alpha, reg, urutan fitur, dst)

Jalankan:
  python prepare_artifacts.py
"""

import sys
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

CONFIG_DIR = Path("/mnt/e/kebutuhan-skripsi/kebutuhan_skripsi_finalz/file-training/config")
sys.path.insert(0, str(CONFIG_DIR))
import paths as P

OUT_DIR = Path("./deploy_bundle")
N_DEMO_USERS = 1000
SEED = 42
ALPHA = 1.0
REG = 0.1
FEATURE_ORDER = [
    "als", "content", "log_pop", "mean_score",
    "wilson", "bayes_wr", "is_adult", "year", "type",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Memuat artefak lengkap...")
    UF = np.load(P.ART_DIR / "als" / "user_factors.npy")
    IF = np.load(P.ART_DIR / "als" / "item_factors.npy")

    n_users_total = UF.shape[0]
    picked = rng.choice(n_users_total, size=min(N_DEMO_USERS, n_users_total), replace=False)
    picked = np.sort(picked)

    np.save(OUT_DIR / "user_factors_demo.npy", UF[picked].astype(np.float32))
    np.save(OUT_DIR / "user_ids_demo.npy", picked.astype(np.int64))
    np.save(OUT_DIR / "item_factors.npy", IF.astype(np.float32))
    print(f"  user_factors_demo : {len(picked)} pengguna")
    print(f"  item_factors      : {IF.shape}")

    print("Menyalin matriks konten...")
    shutil.copy(
        P.ART_DIR / "content" / "item_content_matrix.npz",
        OUT_DIR / "item_content_matrix.npz",
    )

    print("Menyusun fitur item...")
    st = pl.read_parquet(P.PREP_DIR / "train_item_stats.parquet")
    im = pl.read_parquet(P.PREP_DIR / "item_id_map.parquet")
    det = pl.read_parquet(P.PREP_DIR / "details_clean.parquet")
    dm = im.join(det, left_on="anime_id", right_on="mal_id", how="left").sort("item_idx")

    item_features = (
        dm.select(["item_idx", "is_adult", "year_final", "type"])
        .rename({"year_final": "year"})
        .join(
            st.select(["item_idx", "n_train_inter", "mean_score", "wilson_lb", "bayes_wr"]),
            on="item_idx", how="left",
        )
        .with_columns(
            (pl.col("n_train_inter").fill_null(0) + 1).log().alias("log_pop"),
            pl.col("is_adult").fill_null(False).cast(pl.Int8),
            pl.col("year").fill_null(0),
            pl.col("type").fill_null("Unknown"),
            pl.col("mean_score").fill_null(0.0),
            pl.col("wilson_lb").fill_null(0.0).alias("wilson"),
            pl.col("bayes_wr").fill_null(0.0),
        )
        .select(["item_idx", "log_pop", "mean_score", "wilson", "bayes_wr", "is_adult", "year", "type"])
    )
    item_features.write_parquet(OUT_DIR / "item_features.parquet")
    print(f"  item_features : {item_features.height} baris")

    titles = dm.select(["item_idx", "title", "is_adult"]).unique(subset=["item_idx"])
    titles.write_parquet(OUT_DIR / "titles.parquet")
    print(f"  titles : {titles.height} baris")

    print("Menyusun daftar genre per item...")
    genres = dm.select(["item_idx", "genres"]).unique(subset=["item_idx"])
    genres.write_parquet(OUT_DIR / "item_genres.parquet")
    print(f"  item_genres : {genres.height} baris")

    print("Menyusun ground truth demo (test_A untuk 1000 pengguna terpilih)...")
    test_a = pl.read_parquet(P.PREP_DIR / "test_A.parquet", columns=["user_idx", "item_idx", "score"])
    picked_set = set(int(u) for u in picked)
    test_demo = test_a.filter(pl.col("user_idx").is_in(list(picked_set)))
    test_demo.write_parquet(OUT_DIR / "test_relevance_demo.parquet")
    print(f"  test_relevance_demo : {test_demo.height} baris, {test_demo['user_idx'].n_unique()} pengguna")

    print("Menyalin riwayat interaksi (train items per pengguna demo, untuk exclude & profil konten)...")
    tr = pl.read_parquet(P.PREP_DIR / "train.parquet", columns=["user_idx", "item_idx", "is_pos"])
    tr_demo = tr.filter(pl.col("user_idx").is_in(list(picked_set)))
    history = (
        tr_demo.group_by("user_idx")
        .agg([
            pl.col("item_idx").alias("train_items"),
            pl.col("item_idx").filter(pl.col("is_pos")).alias("liked_items"),
        ])
    )
    history.write_parquet(OUT_DIR / "user_history_demo.parquet")
    print(f"  user_history_demo : {history.height} baris")

    print("Menyalin model reranker...")
    shutil.copy(P.ART_DIR / "reranker" / "reranker_yetirank_k100.cbm", OUT_DIR / "reranker.cbm")

    config = {
        "alpha": ALPHA,
        "reg": REG,
        "n_candidates": 100,
        "top_k": 10,
        "max_liked": 100,
        "default_seed_score": 9.0,
        "feature_order": FEATURE_ORDER,
        "n_demo_users": int(len(picked)),
        "n_items": int(IF.shape[0]),
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Selesai. Paket tersimpan di: {OUT_DIR.resolve()}")
    total_mb = sum(f.stat().st_size for f in OUT_DIR.iterdir()) / (1024 * 1024)
    print(f"Ukuran total paket: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()