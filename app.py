import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('novels.csv')

df = load_data()

# Cache TF-IDF similarity model
@st.cache_resource
def compute_similarity(df_clean):
    tfidf_genre = TfidfVectorizer()
    tfidf_subgenre = TfidfVectorizer()

    genre_tfidf = tfidf_genre.fit_transform(df_clean['genre'].astype(str))
    subgenre_tfidf = tfidf_subgenre.fit_transform(df_clean['subgenre'].astype(str))

    genre_sim = cosine_similarity(genre_tfidf)
    subgenre_sim = cosine_similarity(subgenre_tfidf)

    return genre_sim, subgenre_sim

# Session state untuk riwayat
if 'history' not in st.session_state:
    st.session_state.history = []

# Styling
st.markdown("""
<style>
h1, h2, h3 {
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
</style>
""", unsafe_allow_html=True)

# Sidebar Navigasi
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Home", "⭐ Rekomendasi Scored", "🎯 Rekomendasi Genre", "🎭 Rekomendasi TF-IDF", "📊 Distribusi Novel"
])

# -------------------- Home --------------------
if page == "🏠 Home":
    st.title("📚 Daftar Novel Populer")
    top_novels = df.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_novels[['title', 'authors', 'genres', 'scored', 'popularty']], use_container_width=True)

    st.markdown("---")
    st.subheader("📜 Riwayat Rekomendasi")
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            st.markdown(f"### 🔎 Berdasarkan: <code>{item['judul_dipilih']}</code>", unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi.")

# -------------------- Scored --------------------
elif page == "⭐ Rekomendasi Scored":
    st.title("⭐ Rekomendasi Berdasarkan Scored")

    input_score = st.slider("🎯 Pilih Skor:", float(df['scored'].min()), float(df['scored'].max()), float(df['scored'].mean()), step=0.01)

    X = df[['scored']]
    y = df['popularty']
    model = RandomForestRegressor()
    model.fit(X, y)

    df['scored_diff'] = abs(df['scored'] - input_score)
    result = df.sort_values(by='scored_diff').head(5)

    st.dataframe(result[['title', 'authors', 'genres', 'scored']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Scored {input_score:.2f}',
        'rekomendasi': result[['title', 'authors', 'genres', 'scored']]
    })

# -------------------- Genre --------------------
elif page == "🎯 Rekomendasi Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre Novel")

    input_title = st.text_input("✏️ Masukkan Judul Novel (case-sensitive)")
    if input_title:
        selected = df[df['title'] == input_title]
        if not selected.empty:
            genre = selected.iloc[0]['genres']
            result = df[df['genres'] == genre].sort_values(by='scored', ascending=False).head(5)

            st.dataframe(result[['title', 'authors', 'genres', 'scored']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': input_title,
                'rekomendasi': result[['title', 'authors', 'genres', 'scored']]
            })
        else:
            st.warning("Judul tidak ditemukan.")

# -------------------- TF-IDF --------------------
elif page == "🎭 Rekomendasi TF-IDF":
    st.title("🎭 Rekomendasi Berdasarkan Genre & Subgenre")

    input_title = st.text_input("✏️ Masukkan Judul Novel (case-sensitive)")
    if input_title:
        if input_title in df['title'].values:
            index = df[df['title'] == input_title].index[0]
            genre_sim, subgenre_sim = compute_similarity(df)

            sim_score = (genre_sim[index] + subgenre_sim[index]) / 2
            df['similarity'] = sim_score
            recommendations = df.sort_values(by='similarity', ascending=False).drop(index).head(5)

            st.dataframe(recommendations[['title', 'authors', 'genre', 'subgenre', 'scored']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': input_title,
                'rekomendasi': recommendations[['title', 'authors', 'genre', 'subgenre', 'scored']]
            })
        else:
            st.warning("Judul tidak ditemukan.")

# -------------------- Distribusi --------------------
elif page == "📊 Distribusi Novel":
    st.title("📊 Distribusi Genre dan Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Genre")
        genre_counts = df['genres'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        if 'status' in df.columns:
            st.subheader("📘 Status")
            status_counts = df['status'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
            ax2.axis('equal')
            st.pyplot(fig2)
        else:
            st.warning("Kolom 'status' tidak ditemukan.")
