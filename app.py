import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('novels.csv')
    return df

df = load_data()

# Fungsi TF-IDF dan Cosine Similarity
@st.cache_resource
def train_model(df_clean):
    tfidf_genre = TfidfVectorizer()
    tfidf_subgenre = TfidfVectorizer()
    
    genre_tfidf = tfidf_genre.fit_transform(df_clean['genre'].astype(str))
    subgenre_tfidf = tfidf_subgenre.fit_transform(df_clean['subgenre'].astype(str))

    genre_sim = cosine_similarity(genre_tfidf)
    subgenre_sim = cosine_similarity(subgenre_tfidf)
    
    return genre_sim, subgenre_sim

# Inisialisasi session state untuk riwayat rekomendasi
if 'history' not in st.session_state:
    st.session_state.history = []

# CSS styling
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

# Sidebar navigasi
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Home",
    "⭐ Rekomendasi Scored",
    "🎯 Rekomendasi Genre",
    "🎭 Rekomendasi TF-IDF",
    "📊 Distribusi Novel"
])

# ---------------------- Halaman HOME ----------------------
if page == "🏠 Home":
    st.title("📚 Daftar Novel Populer")
    st.markdown("Berikut adalah daftar **10 novel paling populer** berdasarkan data:")

    top_novels = df.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_novels[['title', 'authors', 'genres', 'scored', 'popularty']], use_container_width=True)

    st.markdown("---")
    st.subheader("📜 Riwayat Rekomendasi")
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            st.markdown(f"### 🔎 Rekomendasi berdasarkan: <span style='color:green'><code>{item['judul_dipilih']}</code></span>", unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi di sidebar.")

# ---------------------- Halaman SCORING ----------------------
elif page == "⭐ Rekomendasi Scored":
    st.title("⭐ Rekomendasi Novel Berdasarkan Scored")
    st.markdown("Masukkan skor dan sistem akan merekomendasikan novel dengan **scored serupa** menggunakan algoritma **Random Forest**.")

    input_score = st.slider("🎯 Pilih Nilai Skor", min_value=float(df['scored'].min()), max_value=float(df['scored'].max()), value=float(df['scored'].mean()), step=0.01)

    X = df[['scored']]
    y = df['popularty']
    model = RandomForestRegressor()
    model.fit(X, y)

    df['scored_diff'] = abs(df['scored'] - input_score)
    recommended = df.sort_values(by='scored_diff').head(5)

    st.markdown(f"### 🔍 Rekomendasi berdasarkan skor: <span style='color:green'><code>{input_score:.2f}</code></span>", unsafe_allow_html=True)
    st.dataframe(recommended[['title', 'authors', 'genres', 'scored']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Scored {input_score:.2f}',
        'metode': 'scored',
        'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
    })

# ---------------------- Halaman GENRE ----------------------
elif page == "🎯 Rekomendasi Genre":
    st.title("🎯 Rekomendasi Novel Berdasarkan Genre dari Judul")
    st.markdown("Masukkan judul novel, dan sistem akan menampilkan rekomendasi novel dengan genre yang sama.")

    title_input = st.text_input("✏️ Masukkan Judul Novel (case-sensitive)")

    if title_input:
        selected_novel = df[df['title'] == title_input]

        if not selected_novel.empty:
            selected_genre = selected_novel.iloc[0]['genres']
            recommended = df[df['genres'] == selected_genre].sort_values(by='scored', ascending=False).head(5)

            st.markdown(f"### 📌 Genre: <span style='color:green'><code>{selected_genre}</code></span>", unsafe_allow_html=True)
            st.dataframe(recommended[['title', 'authors', 'genres', 'scored']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': title_input,
                'metode': 'genre',
                'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
            })
        else:
            st.warning("Judul tidak ditemukan dalam data.")

# ---------------------- Halaman TF-IDF ----------------------
elif page == "🎭 Rekomendasi TF-IDF":
    st.title("🎭 Rekomendasi Berdasarkan Kemiripan Genre & Subgenre")
    st.markdown("Masukkan judul novel, dan sistem akan merekomendasikan novel lain yang memiliki kemiripan genre dan subgenre menggunakan **TF-IDF + Cosine Similarity**.")

    genre_sim, subgenre_sim = train_model(df)

    input_title = st.text_input("✏️ Masukkan Judul Novel (case-sensitive)")

    if input_title:
        if input_title in df['title'].values:
            index = df[df['title'] == input_title].index[0]

            sim_score = (genre_sim[index] + subgenre_sim[index]) / 2
            df['similarity'] = sim_score
            top_recs = df.sort_values(by='similarity', ascending=False).drop(index).head(5)

            st.markdown(f"### 📌 Rekomendasi untuk: <span style='color:green'><code>{input_title}</code></span>", unsafe_allow_html=True)
            st.dataframe(top_recs[['title', 'authors', 'genre', 'subgenre', 'scored']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': input_title,
                'metode': 'TF-IDF genre+subgenre',
                'rekomendasi': top_recs[['title', 'authors', 'genre', 'subgenre', 'scored']]
            })
        else:
            st.warning("Judul tidak ditemukan.")

# ---------------------- Halaman DISTRIBUSI ----------------------
elif page == "📊 Distribusi Novel":
    st.title("📊 Distribusi Genre dan Status Novel")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎭 Distribusi Genre")
        genre_counts = df['genres'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        if 'status' in df.columns:
            st.subheader("📘 Distribusi Status Novel")
            status_counts = df['status'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
            ax2.axis('equal')
            st.pyplot(fig2)
        else:
            st.warning("Kolom 'status' tidak ditemukan dalam dataset.")
