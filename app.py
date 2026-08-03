"""
app.py

Antarmuka web untuk demo sistem rekomendasi anime. Menggunakan backend.py
sebagai mesin rekomendasi. Jalankan dengan:

    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from backend import RecommenderEngine

BUNDLE_DIR = Path("./deploy_bundle")

st.set_page_config(
    page_title="Sistem Rekomendasi Anime",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

h1 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

h2, h3 {
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 20px;
    border-radius: 8px 8px 0 0;
}

.result-card {
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    margin-bottom: 10px;
}

.result-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 4px;
}

.result-meta {
    font-size: 0.85rem;
    opacity: 0.75;
}

.metric-box {
    padding: 12px 16px;
    border-radius: 10px;
    background: rgba(120, 120, 120, 0.08);
    text-align: center;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    engine = RecommenderEngine(BUNDLE_DIR)
    engine.load_model()
    return engine


def render_genre_profile(profile):
    if not profile:
        st.caption("Belum ada data genre yang bisa dianalisis dari pilihan ini.")
        return
    for item in profile:
        st.write(f"{item['genre']}  ({item['persentase']}%)")
        st.progress(min(item["persentase"] / 100, 1.0))


def render_history(history):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total interaksi tercatat", history["total_interaksi"])
    with col2:
        st.metric("Total judul disukai", history["total_disukai"])

    if history["contoh_disukai"]:
        with st.expander(f"Lihat judul yang disukai ({len(history['contoh_disukai'])} ditampilkan)"):
            for h in history["contoh_disukai"]:
                st.write(f"- {h['judul']}")


def render_shap_detail(shap):
    st.caption(f"Nilai dasar model (base value): {shap['base_value']}")
    st.table(shap["fitur"])
    st.caption(
        f"Total dari base value ditambah seluruh kontribusi fitur: {shap['total_rekonstruksi']} "
        f"(harus sama dengan skor akhir di atas, sebagai bukti bahwa kontribusi bersifat aditif)."
    )


def render_results(rows, ndcg=None):
    if not rows:
        st.warning("Tidak ada rekomendasi yang bisa ditampilkan untuk pilihan ini.")
        return

    if ndcg is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-box"><div style="font-size:1.4rem;font-weight:700">{ndcg:.4f}</div>'
                f'<div style="font-size:0.8rem;opacity:0.7">NDCG@10 personal</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-box"><div style="font-size:1.4rem;font-weight:700">{len(rows)}</div>'
                f'<div style="font-size:0.8rem;opacity:0.7">Rekomendasi ditampilkan</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-box"><div style="font-size:1.4rem;font-weight:700">Top-10</div>'
                f'<div style="font-size:0.8rem;opacity:0.7">Berdasarkan skor reranker</div></div>',
                unsafe_allow_html=True,
            )
        st.write("")

    left, right = st.columns(2)
    columns = [left, right]
    for i, row in enumerate(rows):
        with columns[i % 2]:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{row['peringkat']}. {row['judul']}</div>
                    <div class="result-meta">
                        skor als {row['skor_als']} &middot;
                        skor konten {row['skor_konten']} &middot;
                        skor akhir {row['skor_akhir']}
                    </div>
                    <div class="result-meta">pendorong utama: {row['pendorong']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"Lihat rincian skor ({len(row['shap']['fitur'])} fitur)"):
                render_shap_detail(row["shap"])


def tab_existing_user(engine):
    st.subheader("Rekomendasi untuk pengguna terdaftar")
    st.write(
        "Mode ini memilih salah satu dari seribu pengguna yang tersedia dalam "
        "data demo. Vektor preferensinya sudah dipelajari selama pelatihan model, "
        "sehingga NDCG personal dapat dihitung karena tersedia data uji sebagai "
        "pembanding."
    )

    user_idx = st.selectbox(
        "Pilih pengguna",
        options=engine.demo_user_ids,
        format_func=lambda u: f"Pengguna #{u}",
    )

    history = engine.get_user_history(user_idx)
    st.markdown("#### Riwayat pengguna")
    render_history(history)

    liked_ids = engine.liked_items_by_user.get(user_idx, [])
    genre_prof = engine.genre_profile(liked_ids)
    st.markdown("#### Profil genre (berdasarkan judul yang disukai)")
    render_genre_profile(genre_prof)

    st.markdown("#### Hasil rekomendasi")
    hide_adult = st.checkbox("Sembunyikan konten dewasa", value=True, key="hide_adult_existing")

    if st.button("Tampilkan rekomendasi", key="btn_existing"):
        with st.spinner("Menghitung rekomendasi..."):
            rows, ndcg = engine.recommend_existing_user(user_idx, hide_adult=hide_adult)
        render_results(rows, ndcg=ndcg)


def tab_guest(engine):
    st.subheader("Rekomendasi untuk profil baru")
    st.write(
        "Mode ini mensimulasikan pengguna baru yang belum pernah tercatat dalam "
        "data pelatihan. Pilih beberapa anime yang disukai sebagai titik awal, "
        "sistem akan membentuk profil sementara melalui fold-in lalu menghasilkan "
        "rekomendasi. Karena profil ini tidak memiliki data uji pembanding, NDCG "
        "tidak dapat dihitung untuk mode ini."
    )

    query = st.text_input("Cari judul anime", placeholder="misal: Naruto")
    matches = engine.search_titles(query, limit=15)
    options = {f"{title}": item_idx for item_idx, title in matches}

    selected_titles = st.multiselect(
        "Pilih anime yang disukai (2 sampai 5 judul)",
        options=list(options.keys()),
    )

    seeds = []
    if selected_titles:
        st.write("Atur tingkat suka untuk setiap judul (opsional, standar 9)")
        for title in selected_titles:
            score = st.slider(title, min_value=1, max_value=10, value=9, key=f"score_{title}")
            seeds.append((options[title], float(score)))

        seed_ids = [item_idx for item_idx, _ in seeds]
        genre_prof = engine.genre_profile(seed_ids)
        st.markdown("#### Profil genre dari pilihan Anda")
        render_genre_profile(genre_prof)

    hide_adult = st.checkbox("Sembunyikan konten dewasa", value=True, key="hide_adult_guest")

    if st.button("Tampilkan rekomendasi", key="btn_guest"):
        if len(seeds) < 2:
            st.warning("Pilih minimal dua anime terlebih dahulu.")
        else:
            with st.spinner("Menghitung rekomendasi..."):
                rows = engine.recommend_guest(seeds, hide_adult=hide_adult)
            render_results(rows, ndcg=None)


def main():
    st.title("Sistem Rekomendasi Anime")
    st.write(
        "Demo ini menampilkan hasil model hybrid yang menggabungkan Collaborative "
        "Filtering, Content-Based Filtering, dan pemeringkatan ulang berbasis "
        "CatBoost."
    )

    engine = load_engine()

    tab1, tab2 = st.tabs(["Pengguna terdaftar", "Profil baru"])
    with tab1:
        tab_existing_user(engine)
    with tab2:
        tab_guest(engine)


if __name__ == "__main__":
    main()