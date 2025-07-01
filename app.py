import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    return pd.read_csv('novels_selected.csv')

df = load_data()

# ------------------ Standardisasi Nama Kolom ------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ------------------ Isi NaN/None dengan Nilai Terdekat ------------------
cols_to_fill = ['score', 'volume', 'chapter', 'years_finish']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

# ------------------ Inisialisasi Session State ------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------ Ekstraksi Fitur TF-IDF ------------------
df['title'] = df['title'].fillna('').str.lower().str.replace('[^a-zA-Z]', ' ', regex=True).str.replace('\s+', ' ', regex=True).str.strip()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])

# ------------------ CSS & Background ------------------
st.markdown("""
<style>
h1, h2, h3 {
    color: #2E8B57;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}
.stButton>button {
    color: white;
    background-color: #2E8B57;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stDataFrame {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Sidebar Navigasi ------------------
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Rekomendasi Score", "Rekomendasi Genre", "Distribusi Novel"])

# ------------------ Halaman Home ------------------
if page == "Home":
    st.image("https://cdn-icons-png.flaticon.com/512/29/29302.png", width=100)
    st.title("📚 Daftar Novel Populer")
    st.markdown("Selamat datang di aplikasi rekomendasi novel! 📖✨")

    st.markdown("### 🏆 10 Novel Paling Populer")
    top_novels = df.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_novels[['title', 'author','type', 'genre', 'score', 'popularty']], use_container_width=True)

    st.markdown("### ⭐ 10 Novel dengan Rating Tertinggi")
    top_rated_novels = df.sort_values(by="score", ascending=False).head(10)
    st.dataframe(top_rated_novels[['title', 'author','type', 'genre', 'score', 'popularty']], use_container_width=True)

    st.markdown("---")
    st.subheader("📜 Riwayat Rekomendasi")
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            st.markdown(f"### 🔎 Rekomendasi berdasarkan: <span style='color:green'><code>{item['judul_dipilih']}</code></span>", unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi di sidebar.")

# ------------------ Halaman Rekomendasi Score ------------------
elif page == "Rekomendasi Score":
    st.image("https://cdn-icons-png.flaticon.com/512/2866/2866327.png", width=100)
    st.title("📈 Rekomendasi Berdasarkan Score")
    st.markdown("Temukan novel yang direkomendasikan berdasarkan **skor pilihanmu** dan kemiripan judul. 🎯📘")

    input_score = st.slider("Pilih Skor", min_value=float(df['score'].min()), max_value=float(df['score'].max()), value=float(df['score'].mean()), step=0.01)

    X = df[['score']]
    y = df['popularty']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    r2 = model.score(X, y)
    predicted_pop = model.predict([[input_score]])[0]
    df['predicted_popularty'] = model.predict(X)
    df['predicted_diff'] = abs(df['predicted_popularty'] - predicted_pop)

    tfidf_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    input_idx = np.argmin(abs(df['score'] - input_score))
    df['tfidf_sim'] = tfidf_similarities[input_idx]
    df['combined_score'] = df['tfidf_sim'] - df['predicted_diff'] / df['predicted_diff'].max()

    recommended = df.sort_values(by='combined_score', ascending=False).head(5)

    st.markdown(f"🔎 <b>Model R² Score:</b> <code>{r2:.4f}</code>", unsafe_allow_html=True)
    st.markdown(f"📊 <b>Prediksi Popularitas untuk skor {input_score:.2f}:</b> <code>{predicted_pop:.2f}</code>", unsafe_allow_html=True)
    st.dataframe(recommended[['title', 'author', 'type','genre', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Score {input_score:.2f}',
        'metode': 'random_forest + tfidf_similarity',
        'rekomendasi': recommended[['title', 'author','type', 'genre', 'score']]
    })

# ------------------ Halaman Rekomendasi Genre ------------------
elif page == "Rekomendasi Genre":
    st.image("https://cdn-icons-png.flaticon.com/512/2165/2165006.png", width=100)
    st.title("🎯 Rekomendasi Berdasarkan Genre & Judul")
    st.markdown("Masukkan judul novel favoritmu, dan sistem akan mencari novel sejenis berdasarkan genre dan prediksi popularitas. 🔍📖")

    title_input = st.text_input("Masukkan Judul Novel")

    if title_input:
        matched = df[df['title'].str.contains(title_input, case=False, na=False)]
        if not matched.empty:
            selected_genre = matched.iloc[0]['genre']
            genre_novels = df[df['genre'] == selected_genre].copy()

            Xg = genre_novels[['score']]
            yg = genre_novels['popularty']
            model_g = RandomForestRegressor(n_estimators=100, random_state=42)
            model_g.fit(Xg, yg)
            genre_novels['predicted_popularty'] = model_g.predict(Xg)

            genre_idx = genre_novels.index
            title_vec = tfidf_vectorizer.transform([title_input])
            tfidf_sim = cosine_similarity(title_vec, tfidf_matrix[genre_idx]).flatten()
            genre_novels['tfidf_sim'] = tfidf_sim
            genre_novels['combined_score'] = genre_novels['tfidf_sim'] + genre_novels['predicted_popularty'] / genre_novels['predicted_popularty'].max()

            recommended = genre_novels.sort_values(by='combined_score', ascending=False).head(5)
            r2g = model_g.score(Xg, yg)

            st.markdown(f"<b>📚 Genre:</b> <code>{selected_genre}</code> | <b>Model R²:</b> <code>{r2g:.4f}</code>", unsafe_allow_html=True)
            st.dataframe(recommended[['title', 'author', 'type','genre', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': title_input,
                'metode': 'genre + tfidf + random_forest',
                'rekomendasi': recommended[['title', 'author','type', 'genre', 'score']]
            })
        else:
            st.warning("⚠️ Judul tidak ditemukan dalam data.")

# ------------------ Halaman Distribusi ------------------
elif page == "Distribusi Novel":
    st.image("https://cdn-icons-png.flaticon.com/512/3502/3502458.png", width=100)
    st.title("📊 Distribusi Novel")
    st.markdown("Visualisasi statistik seputar genre, status, dan tahun rilis novel. 📈")

    st.markdown("### 🎭 Top 10 Genre")
    genre_counts = df['genre'].value_counts().head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(genre_counts.index, genre_counts.values, color='#3CB371')
    ax1.set_ylabel("Jumlah Novel")
    ax1.set_title("Distribusi 10 Genre Terbanyak")
    ax1.set_xticklabels(genre_counts.index, rotation=45, ha='right')
    st.pyplot(fig1)

    if 'status' in df.columns:
        st.markdown("### 📘 Status Penerbitan")
        status_counts = df['status'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', colors=['#87CEFA', '#32CD32'])
        ax2.set_title("Distribusi Status Novel")
        st.pyplot(fig2)

    if 'years_start' in df.columns:
        st.markdown("### 🕰️ Tahun Mulai")
        ys = df['years_start'].dropna().astype(int).value_counts().sort_index()
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(ys.index, ys.values, marker='o', color='#1E90FF')
        ax3.set_xlabel("Tahun")
        ax3.set_ylabel("Jumlah Novel")
        ax3.set_title("Distribusi Tahun Mulai")
        st.pyplot(fig3)

    if 'years_finish' in df.columns:
        st.markdown("### ⏳ Tahun Selesai")
        yf = df['years_finish'].dropna().astype(int).value_counts().sort_index()
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(yf.index, yf.values, marker='s', linestyle='--', color='#FFA500')
        ax4.set_xlabel("Tahun")
        ax4.set_ylabel("Jumlah Novel")
        ax4.set_title("Distribusi Tahun Selesai")
        st.pyplot(fig4)
