import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    return pd.read_csv('novels_selected.csv')  # gunakan path file yang diupload

df = load_data()

# Inisialisasi session state untuk menyimpan riwayat
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------ CSS Styling ------------------
st.markdown("""
<style>
h1, h2, h3, h4 {
    color: #2E8B57;
}
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}
.stButton>button {
    color: white;
    background-color: #2E8B57;
    border-radius: 10px;
}
.stTable {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------ Sidebar ------------------
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Home", "⭐ Rekomendasi Score", "🎯 Rekomendasi Genre", "📊 Distribusi Novel"])

# ---------------------- Home Page ----------------------
if page == "🏠 Home":
    st.title("📚 Daftar Novel Populer")
    st.markdown("Berikut adalah daftar **10 novel paling populer** berdasarkan data:")

    top_novels = df.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_novels[['title', 'author', 'genre', 'score', 'popularty']], use_container_width=True)

    st.markdown("### 🏅 10 Novel dengan Rating Tertinggi")
    top_rated_novels = df.sort_values(by="score", ascending=False).head(10)
    st.dataframe(top_rated_novels[['title', 'author', 'genre', 'score', 'popularty']], use_container_width=True)

    st.markdown("---")
    st.subheader("📜 Riwayat Rekomendasi")
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            st.markdown(f"### 🔎 Rekomendasi berdasarkan: <span style='color:green'><code>{item['judul_dipilih']}</code></span>", unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi di sidebar.")

# ------------------ Rekomendasi Berdasarkan Scored ------------------
elif page == "⭐ Rekomendasi Score":
    st.title("⭐ Rekomendasi Novel Berdasarkan Score")
    st.markdown("Masukkan skor dan sistem akan merekomendasikan novel dengan **score serupa** menggunakan algoritma **Random Forest Regressor**.")

    input_score = st.slider("🎯 Pilih Nilai Skor", min_value=float(df['score'].min()),
                            max_value=float(df['score'].max()), 
                            value=float(df['score'].mean()), step=0.01)

    # Pelatihan model
    X = df[['score']]
    y = df['popularty']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Evaluasi model
    r2_score = model.score(X, y)
    st.markdown(f"📈 <b>Model R² Score:</b> <code>{r2_score:.4f}</code>", unsafe_allow_html=True)

    # Prediksi popularitas
    predicted_pop = model.predict([[input_score]])[0]
    st.markdown(f"📊 <b>Prediksi Popularitas untuk skor {input_score:.2f}:</b> <code>{predicted_pop:.2f}</code>", unsafe_allow_html=True)

    # Tambahkan prediksi ke data
    df['predicted_popularty'] = model.predict(df[['score']])
    df['predicted_diff'] = abs(df['predicted_popularty'] - predicted_pop)
    recommended = df.sort_values(by='predicted_diff').head(5)

    st.markdown("### 📚 Rekomendasi Novel:")
    st.dataframe(recommended[['title', 'author', 'genre', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Score {input_score:.2f}',
        'metode': 'random_forest',
        'rekomendasi': recommended[['title', 'author', 'genre', 'score']]
    })

# ------------------ Rekomendasi Berdasarkan Genre & Judul Serupa + Random Forest ------------------
elif page == "🎯 Rekomendasi Genre":
    st.title("🎯 Rekomendasi Novel Berdasarkan Genre & Judul Serupa")
    st.markdown("Masukkan judul novel, dan sistem akan menampilkan rekomendasi novel dengan **genre yang sama** dan **judul yang mirip**, menggunakan prediksi popularitas dari algoritma **Random Forest**.")

    title_input = st.text_input("✏️ Masukkan Judul Novel (boleh sebagian)")

    if title_input:
        # Temukan judul yang mengandung teks input
        matched_titles = df[df['title'].str.contains(title_input, case=False, na=False)]

        if not matched_titles.empty:
            # Gunakan genre dari judul pertama yang cocok
            selected_genre = matched_titles.iloc[0]['genre']
            st.markdown(f"### 📌 Genre Ditemukan: <span style='color:green'><code>{selected_genre}</code></span>", unsafe_allow_html=True)

            # Filter semua novel dengan genre yang sama
            genre_novels = df[df['genre'] == selected_genre].copy()

            # Latih model Random Forest pada genre ini
            X_genre = genre_novels[['score']]
            y_genre = genre_novels['popularty']
            model_genre = RandomForestRegressor(n_estimators=100, random_state=42)
            model_genre.fit(X_genre, y_genre)

            # Prediksi popularitas untuk seluruh novel dalam genre
            genre_novels['predicted_popularty'] = model_genre.predict(X_genre)

            # Filter novel dengan judul mirip
            similar_titles = genre_novels[genre_novels['title'].str.contains(title_input, case=False, na=False)]

            # Jika tidak cukup, tambahkan dari genre yang sama
            if len(similar_titles) < 5:
                additional = genre_novels[~genre_novels['title'].str.contains(title_input, case=False, na=False)]
                similar_titles = pd.concat([similar_titles, additional]).drop_duplicates()

            # Ambil 5 teratas berdasarkan score tertinggi
            recommended = similar_titles.sort_values(by='score', ascending=False).head(5)

            # Evaluasi model
            r2_genre = model_genre.score(X_genre, y_genre)
            st.markdown(f"📈 <b>Model R² Score (genre ini):</b> <code>{r2_genre:.4f}</code>", unsafe_allow_html=True)

            # Tampilkan hasil
            st.markdown("### 📚 Rekomendasi Novel:")
            st.dataframe(recommended[['title', 'author', 'genre', 'type', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': title_input,
                'metode': 'genre + judul mirip + random_forest',
                'rekomendasi': recommended[['title', 'author', 'genre', 'type', 'score']]
            })

        else:
            st.warning("Judul tidak ditemukan dalam data.")




# ------------------ Distribusi Genre dan Status ------------------
elif page == "📊 Distribusi Novel":
    st.title("📊 Distribusi Novel Berdasarkan Genre, Status, dan Tahun")

    st.markdown("### 📘 Distribusi 10 Genre Terpopuler")
    genre_counts = df['genre'].value_counts().head(10)
    fig_genre, ax_genre = plt.subplots()
    ax_genre.bar(genre_counts.index, genre_counts.values, color='skyblue')
    ax_genre.set_ylabel("Jumlah Novel")
    ax_genre.set_xlabel("Genre")
    ax_genre.set_title("Top 10 Genre Novel")
    ax_genre.tick_params(axis='x', rotation=45)
    st.pyplot(fig_genre)

    st.markdown("### 📗 Distribusi Status Novel")
    if 'status' in df.columns:
        status_counts = df['status'].value_counts()
        fig_status, ax_status = plt.subplots()
        ax_status.bar(status_counts.index, status_counts.values, color='lightgreen')
        ax_status.set_ylabel("Jumlah Novel")
        ax_status.set_xlabel("Status")
        ax_status.set_title("Distribusi Status Novel")
        st.pyplot(fig_status)
    else:
        st.warning("Kolom 'status' tidak ditemukan dalam dataset.")

    st.markdown("### 📆 Distribusi Tahun Mulai Novel")
    if 'years start' in df.columns:
        year_start_counts = df['years start'].dropna().astype(int).value_counts().sort_index()
        fig_start, ax_start = plt.subplots()
        ax_start.plot(year_start_counts.index, year_start_counts.values, marker='o', linestyle='-')
        ax_start.set_ylabel("Jumlah Novel")
        ax_start.set_xlabel("Tahun Mulai")
        ax_start.set_title("Distribusi Tahun Mulai Novel")
        st.pyplot(fig_start)

    st.markdown("### 📅 Distribusi Tahun Selesai Novel")
    if 'years finish' in df.columns:
        year_finish_counts = df['years finish'].dropna().astype(int).value_counts().sort_index()
        fig_finish, ax_finish = plt.subplots()
        ax_finish.plot(year_finish_counts.index, year_finish_counts.values, marker='s', linestyle='--', color='orange')
        ax_finish.set_ylabel("Jumlah Novel")
        ax_finish.set_xlabel("Tahun Selesai")
        ax_finish.set_title("Distribusi Tahun Selesai Novel")
        st.pyplot(fig_finish)
