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

# ------------------ Isi NaN/None dengan Nilai Terdekat (Kolom Tertentu) ------------------
cols_to_fill = ['score', 'volume', 'chapter', 'years_finish']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

# ------------------ Cek Duplikat ------------------
jumlah_duplikat = df.duplicated().sum()

# ------------------ Ukuran DataFrame ------------------
novel_rows, novel_cols = df.shape

# ------------------ Inisialisasi Session State ------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------ Ekstraksi Fitur TF-IDF ------------------
df['title'] = df['title'].fillna('').str.lower().str.replace('[^a-zA-Z]', ' ', regex=True).str.replace('\s+', ' ', regex=True).str.strip()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])

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

# ------------------ Sidebar Navigasi ------------------
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Rekomendasi Score", "Rekomendasi Genre", "Distribusi Novel"])

# ---------------------- Home ----------------------
if page == "Home":
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
            st.markdown(f"### 🔎 Rekomendasi berdasarkan: <span style='color:green'><code>{item['judul_dipilih']}</code></span>", unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi di sidebar.")

# ------------------ Rekomendasi Score ------------------
elif page == "Rekomendasi Score":
    st.title("Rekomendasi Berdasarkan Score")
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

    st.markdown(f"<b>Model R² Score:</b> <code>{r2:.4f}</code>", unsafe_allow_html=True)
    st.markdown(f"<b>Prediksi Popularitas untuk skor {input_score:.2f}:</b> <code>{predicted_pop:.2f}</code>", unsafe_allow_html=True)
    st.dataframe(recommended[['title', 'author', 'type','genre', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Score {input_score:.2f}',
        'metode': 'random_forest + tfidf_similarity',
        'rekomendasi': recommended[['title', 'author','type', 'genre', 'score']]
    })

# ------------------ Rekomendasi Genre ------------------
elif page == "Rekomendasi Genre":
    st.title("Rekomendasi Berdasarkan Genre & Judul")
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

            st.markdown(f"<b>Genre:</b> <code>{selected_genre}</code> | <b>Model R²:</b> <code>{r2g:.4f}</code>", unsafe_allow_html=True)
            st.dataframe(recommended[['title', 'author', 'type','genre', 'score', 'popularty', 'predicted_popularty']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': title_input,
                'metode': 'genre + tfidf + random_forest',
                'rekomendasi': recommended[['title', 'author','type', 'genre', 'score']]
            })
        else:
            st.warning("Judul tidak ditemukan dalam data.")

# ------------------ Distribusi Novel ------------------
elif page == "Distribusi Novel":
    st.title("Distribusi Novel")

    st.markdown("### Top 10 Genre")
    genre_counts = df['genre'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    ax1.bar(genre_counts.index, genre_counts.values, color='skyblue')
    ax1.set_xticklabels(genre_counts.index, rotation=45)
    st.pyplot(fig1)

    if 'status' in df.columns:
        st.markdown("### Distribusi Status")
        status_counts = df['status'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.bar(status_counts.index, status_counts.values, color='lightgreen')
        st.pyplot(fig2)

    if 'years_start' in df.columns:
        st.markdown("### Tahun Mulai")
        ys = df['years_start'].dropna().astype(int).value_counts().sort_index()
        fig3, ax3 = plt.subplots()
        ax3.plot(ys.index, ys.values, marker='o')
        st.pyplot(fig3)

    if 'years_finish' in df.columns:
        st.markdown("### Tahun Selesai")
        yf = df['years_finish'].dropna().astype(int).value_counts().sort_index()
        fig4, ax4 = plt.subplots()
        ax4.plot(yf.index, yf.values, marker='s', linestyle='--', color='orange')
        st.pyplot(fig4)
