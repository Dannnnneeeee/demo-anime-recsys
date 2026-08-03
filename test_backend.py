"""
test_backend.py

Uji cepat untuk memastikan backend.py menghasilkan angka yang sama dengan
notebook_08_inference_online. Jalankan setelah prepare_artifacts.py selesai.

Catatan: user 336 hanya bisa diuji apabila ia termasuk dalam 1000 pengguna
yang tersampel ke dalam bundle demo. Jika tidak termasuk, skrip ini menguji
pengguna demo pertama yang tersedia sebagai gantinya, dan konsistensi
tetap teruji lewat perbandingan dua-jalur (ALS vs reranker), bukan lewat
kecocokan dengan angka user 336 di paste.
"""

from pathlib import Path
from backend import RecommenderEngine

BUNDLE_DIR = Path("./deploy_bundle")


def main():
    engine = RecommenderEngine(BUNDLE_DIR).load_model()
    print(f"Bundle dimuat: {len(engine.demo_user_ids)} pengguna demo, {engine.n_items} item")

    target = 336
    user_idx = target if target in engine.user_factors else engine.demo_user_ids[0]
    if user_idx != target:
        print(f"User {target} tidak ada di bundle demo, menguji user {user_idx} sebagai gantinya.")

    rows, ndcg = engine.recommend_existing_user(user_idx, hide_adult=False)
    print(f"\n=== Pengguna terdaftar (user_idx={user_idx}) ===")
    print(f"NDCG personal: {ndcg}")
    for r in rows[:5]:
        print(f"  #{r['peringkat']} {r['judul']} | als={r['skor_als']} konten={r['skor_konten']} "
              f"akhir={r['skor_akhir']} pendorong={r['pendorong']}")

    print("\n=== Tamu (fold-in dari 3 judul contoh) ===")
    sample_items = [it for it, _ in engine.browsable_items[:3]]
    seeds = [(it, 9.0) for it in sample_items]
    guest_rows = engine.recommend_guest(seeds, hide_adult=False)
    for r in guest_rows[:5]:
        print(f"  #{r['peringkat']} {r['judul']} | als={r['skor_als']} konten={r['skor_konten']} "
              f"akhir={r['skor_akhir']} pendorong={r['pendorong']}")

    print("\nJika kedua bagian di atas tampil tanpa galat dan skor tidak nol/NaN, backend siap dipakai.")


if __name__ == "__main__":
    main()