"""
paths.py — Konfigurasi yang TIDAK bergantung pada hasil analisis.

Isi file ini sengaja dibatasi hanya pada:
  - lokasi folder & file
  - konstanta reproducibility (seed, hash salt)
  - definisi yang murni konvensi/skema (mis. score 0 = unscored)

YANG TIDAK BOLEH ADA DI SINI:
  - k-core / min raters / min interaksi
  - m Bayesian, threshold positif, ALPHA, shrinkage, dll
  Semua angka di atas adalah HASIL Fase 1 (EDA) dan ditulis ke
  decisions/eda_decisions.json, lalu dibaca oleh Fase 2 ke atas.

Path WSL:
  Windows  : E:\\kebutuhan-skripsi\\kebutuhan_skripsi_finalz
  WSL/Linux: /mnt/e/kebutuhan-skripsi/kebutuhan_skripsi_finalz
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Lokasi dasar
# ----------------------------------------------------------------------
# BASE = Path("/mnt/e/kebutuhan-skripsi/kebutuhan_skripsi_finalz")
BASE = Path(r"E:\kebutuhan-skripsi\kebutuhan_skripsi_finalz")
DATASET_DIR = BASE / "dataset"        # CSV mentah
TRAIN_DIR   = BASE / "file-training"  # semua output ada di sini

# ----------------------------------------------------------------------
# Output per-fase (tiap fase menulis HANYA ke foldernya sendiri)
# ----------------------------------------------------------------------
PARQUET_DIR   = TRAIN_DIR / "parquet"            # Fase 0
FIG_DIR       = TRAIN_DIR / "outputs" / "figures"  # Fase 1
TBL_DIR       = TRAIN_DIR / "outputs" / "tables"   # Fase 1
DECISIONS_DIR = TRAIN_DIR / "decisions"          # Fase 1 -> kontrak antar-fase
PREP_DIR      = TRAIN_DIR / "prep"               # Fase 2
ART_DIR       = TRAIN_DIR / "artifacts"          # Fase 3-5

# File kontrak antar-fase: ditulis Fase 1, dibaca Fase 2+
EDA_DECISIONS = DECISIONS_DIR / "eda_decisions.json"

# ----------------------------------------------------------------------
# Nama file CSV mentah  (favs = future work, tidak dipakai)
# ----------------------------------------------------------------------
CSV = {
    "ratings":  DATASET_DIR / "ratings.csv",
    "details":  DATASET_DIR / "details.csv",
    "stats":    DATASET_DIR / "stats.csv",
    "profiles": DATASET_DIR / "profiles.csv",
}

# Nama file Parquet hasil konversi setia (Fase 0)
PARQUET = {
    "ratings":  PARQUET_DIR / "ratings.parquet",
    "details":  PARQUET_DIR / "details.parquet",
    "stats":    PARQUET_DIR / "stats.parquet",
    "profiles": PARQUET_DIR / "profiles.parquet",
}

# ----------------------------------------------------------------------
# Reproducibility (prinsip: jangan pernah np.random)
# ----------------------------------------------------------------------
SEED = 42
HASH_SALT = "skripsi_anime_recsys_v2"

# ----------------------------------------------------------------------
# Konvensi skema (fakta dataset, bukan keputusan analitis)
# ----------------------------------------------------------------------
SCORE_UNSCORED = 0   # score = 0 berarti BELUM dinilai, bukan rating nol

# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def ensure_dirs():
    """Buat folder output bila belum ada (tidak menyentuh dataset mentah)."""
    for d in (PARQUET_DIR, FIG_DIR, TBL_DIR, DECISIONS_DIR, PREP_DIR, ART_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return True
