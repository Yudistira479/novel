import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    return pd.read_csv('novels_selected.csv')  # gunakan path file yang diupload

df = load_data()

# Inisialisasi session state untuk menyimpan riwayat
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------ Ekstraksi Fitur TF-IDF ------------------
df['title'] = df['title'].fillna('').str.lower().str.replace('[^a-zA-Z]', ' ', regex=True).str.replace('\s+', ' ', regex=True).str.strip()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])
tfidf_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

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
    st.dataframe(top_novels[['title', 'author','type', 'genre', 'score', 'popularty']], use_container_width=True)

    st.markdown("### 🏅 10 Novel dengan Rating Tertinggi")
    top_rated_novels = df.sort_values(by="score", ascending=False).head(10)
    st.dataframe(top_rated_novels[['title', 'author','type', 'genre', 'score', 'popularty']], use_container_width=True)

    st.markdown("---")
    st.subheader("📜 Riwayat Rekomendasi")
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            st.markdown(f"### 🔎 Rekomendasi berdasarkan: <span style='color:green'><code>{item['judul_dipilih']}</code></span>", 
                        unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi di sidebar.")

# ------------------ Rekomendasi Berdasarkan Scored ------------------
elif page == "⭐ Rekomendasi Score":
    st.title("⭐ Rekomendasi Novel Berdasarkan Score")
    st.markdown("Masukkan skor dan sistem akan merekomendasikan novel dengan **score serupa** menggunakan algoritma **Random Forest Regressor** dan kemiripan judul TF-IDF.")

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

    # Rekomendasi berdasarkan prediksi popularitas
    df['predicted_popularty'] = model.predict(df[['score']])
    df['predicted_diff'] = abs(df['predicted_popularty'] - predicted_pop)

    # Kombinasikan dengan TF-IDF similarity
    tfidf_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    input_idx = np.argmin(abs(df['score'] - input_score))
    tfidf_scores = tfidf_similarities[input_idx]
    df['tfidf_sim'] = tfidf_scores

    # Rekomendasi berdasarkan skor gabungan
    df['combined_score'] = df['tfidf_sim'] - df['predicted_diff'] / df['predicted_diff'].max()
    recommended = df.sort_values(by='combined_score', ascending=False).head(5)

    st.markdown("### 📚 Rekomendasi Novel:")
    st.dataframe(recommended[['title', 'author', 'type','genre', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Score {input_score:.2f}',
        'metode': 'random_forest + tfidf_similarity',
        'rekomendasi': recommended[['title', 'author','type', 'genre', 'score']]
    })

# ------------------ Rekomendasi Berdasarkan Genre & Judul Serupa + Random Forest ------------------
elif page == "🎯 Rekomendasi Genre":
    st.title("🎯 Rekomendasi Novel Berdasarkan Genre & Judul Serupa")
    st.markdown("Masukkan judul novel, dan sistem akan menampilkan rekomendasi novel dengan **genre yang sama** dan **judul yang mirip**, menggunakan **Random Forest** dan kemiripan judul **TF-IDF**.")

    title_input = st.text_input("✏️ Masukkan Judul Novel (boleh sebagian)")

    if title_input:
        matched_titles = df[df['title'].str.contains(title_input, case=False, na=False)]

        if not matched_titles.empty:
            selected_genre = matched_titles.iloc[0]['genre']
            st.markdown(f"### 📌 Genre Ditemukan: <span style='color:green'><code>{selected_genre}</code></span>", unsafe_allow_html=True)

            genre_novels = df[df['genre'] == selected_genre].copy()
            X_genre = genre_novels[['score']]
            y_genre = genre_novels['popularty']
            model_genre = RandomForestRegressor(n_estimators=100, random_state=42)
            model_genre.fit(X_genre, y_genre)

            genre_novels['predicted_popularty'] = model_genre.predict(X_genre)

            # Kemiripan TF-IDF dalam genre
            genre_indices = genre_novels.index
            title_vector = tfidf_vectorizer.transform([title_input])
            genre_tfidf = tfidf_matrix[genre_indices]
            tfidf_sim = cosine_similarity(title_vector, genre_tfidf).flatten()
            genre_novels['tfidf_sim'] = tfidf_sim

            genre_novels['combined_score'] = genre_novels['tfidf_sim'] + genre_novels['predicted_popularty'] / genre_novels['predicted_popularty'].max()

            recommended = genre_novels.sort_values(by='combined_score', ascending=False).head(5)

            r2_genre = model_genre.score(X_genre, y_genre)
            st.markdown(f"📈 <b>Model R² Score (genre ini):</b> <code>{r2_genre:.4f}</code>", unsafe_allow_html=True)

            st.markdown("### 📚 Rekomendasi Novel:")
            st.dataframe(recommended[['title', 'author','type', 'genre', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': title_input,
                'metode': 'genre + tfidf + random_forest',
                'rekomendasi': recommended[['title', 'author', 'type','genre', 'score']]
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
