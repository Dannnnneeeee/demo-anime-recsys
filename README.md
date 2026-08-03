# Demo Sistem Rekomendasi Anime

## Struktur

- `prepare_artifacts.py` : membuat paket artefak kecil dari artefak lengkap penelitian, dijalankan sekali secara lokal di komputer yang menyimpan hasil training.
- `backend.py` : mesin rekomendasi, replika langsung dari notebook 08 (confidence, fold-in, skor konten, reranking, penjelasan SHAP).
- `test_backend.py` : uji cepat backend tanpa antarmuka web.
- `app.py` : antarmuka Streamlit.
- `requirements.txt` : daftar pustaka yang dibutuhkan.

## Langkah menjalankan

1. Pastikan `paths.py` pada proyek penelitian sudah mengarah ke lokasi artefak yang benar.

2. Buat paket artefak kecil (dijalankan sekali, di komputer yang menyimpan artefak lengkap):

   ```
   python prepare_artifacts.py
   ```

   Perintah ini menghasilkan folder `deploy_bundle` berisi seluruh berkas yang dibutuhkan aplikasi, termasuk 1000 pengguna terpilih untuk mode "pengguna terdaftar".

3. Uji backend tanpa antarmuka web terlebih dahulu:

   ```
   python test_backend.py
   ```

   Pastikan tidak ada galat dan skor yang tampil bukan nol atau NaN.

4. Pasang dependensi dan jalankan aplikasi:

   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Untuk deploy ke Streamlit Community Cloud, unggah folder ini beserta `deploy_bundle` ke repositori, lalu hubungkan repositori tersebut di streamlit.io. Folder `deploy_bundle` sebaiknya disertakan langsung di repositori karena ukurannya sudah diperkecil pada langkah 2.

## Catatan konsistensi

Seluruh rumus pada `backend.py` (confidence, fold-in, skor konten, urutan fitur, definisi grade untuk NDCG) disalin persis dari notebook 08 tanpa modifikasi, sehingga hasil pada aplikasi ini konsisten dengan angka yang dilaporkan pada skripsi.
