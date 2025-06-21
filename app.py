import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np
import urllib.request # Untuk mengecek ketersediaan file model dari GitHub jika diperlukan

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Novel Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path dan URL ---
MODEL_PATH = 'model_pelatihan.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
DATA_URL = 'https://github.com/Yudistira479/novel/blob/main/novels.csv'

# --- Fungsi Pemuatan Data dan Model (dengan caching untuk performa) ---

@st.cache_data
def load_data(url):
    """Memuat dataset novel dari URL GitHub."""
    try:
        # Menambahkan on_bad_lines='skip' untuk melewati baris yang rusak
        # Menambahkan sep=',' dan encoding='utf-8' untuk penanganan CSV yang lebih baik
        df = pd.read_csv(url, sep=',', on_bad_lines='skip', encoding='utf-8')
        df.fillna('', inplace=True)
        required_columns = ['Title', 'Description', 'Genre', 'Status', 'Volume', 'Favorites', 'Views', 'Score', 'Tags']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Kolom '{col}' tidak ditemukan dalam dataset dari GitHub. Pastikan nama kolom sesuai.")
                st.stop() # Menghentikan eksekusi aplikasi jika kolom penting hilang
        
        # Pastikan 'Views' dan 'Score' adalah numerik
        df['Views'] = pd.to_numeric(df['Views'], errors='coerce').fillna(0)
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error saat memuat atau memproses novels.csv dari GitHub: {e}")
        st.stop()

@st.cache_resource
def load_model_and_vectorizer():
    """Memuat model dan vectorizer yang sudah dilatih."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error(f"Model atau TF-IDF Vectorizer tidak ditemukan ({MODEL_PATH}, {VECTORIZER_PATH}).")
        st.warning("Silakan jalankan `model_pelatihan.py` terlebih dahulu untuk melatih dan menyimpan model.")
        st.stop()
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return model, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error saat memuat model atau vectorizer: {e}")
        st.stop()

# Muat data, model, dan vectorizer
novels_df = load_data(DATA_URL)
model, tfidf_vectorizer = load_model_and_vectorizer()

# Siapkan fitur gabungan untuk TF-IDF dan matriks TF-IDF
tfidf_columns = ['Title', 'Description', 'Genre', 'Tags']
novels_df['combined_features'] = novels_df[tfidf_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
tfidf_matrix = tfidf_vectorizer.transform(novels_df['combined_features'])

# Inisialisasi riwayat pencarian di Streamlit's session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

def get_recommendations_content_based(novel_title, top_n=10):
    """
    Memberikan rekomendasi novel berdasarkan kemiripan konten.
    """
    if novel_title not in novels_df['Title'].values:
        return pd.DataFrame()

    idx = novels_df[novels_df['Title'] == novel_title].index[0]
    novel_features = tfidf_matrix[idx]

    cosine_similarities = (tfidf_matrix * novel_features.T).toarray().flatten()
    similar_indices = cosine_similarities.argsort()[::-1]
    
    similar_novels = novels_df.iloc[similar_indices].copy()
    similar_novels['Similarity_Score'] = cosine_similarities[similar_indices]
    
    similar_novels = similar_novels[similar_novels['Title'] != novel_title]
    
    return similar_novels.head(top_n)

def add_to_history(query):
    """Menambahkan query ke riwayat pencarian Streamlit."""
    if query not in st.session_state.search_history:
        st.session_state.search_history.insert(0, query)
    if len(st.session_state.search_history) > 5:
        st.session_state.search_history.pop()

# --- Navigasi Halaman ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Rekomendasi (Novel)", "Rekomendasi (Genre)"])

if page == "Beranda":
    st.title("Selamat Datang di NovelRecommender!")

    col1, col2 = st.columns(2)

    with col1:
        st.header("10 Novel Terpopuler")
        top_novels = novels_df.sort_values(by='Views', ascending=False).head(10)
        if not top_novels.empty:
            st.dataframe(top_novels[['Title', 'Genre', 'Views']].style.format({"Views": "{:,.0f}"}))
        else:
            st.info("Tidak ada novel terpopuler untuk ditampilkan.")

    with col2:
        st.header("Riwayat Pencarian Anda")
        if st.session_state.search_history:
            for item in st.session_state.search_history:
                st.write(f"- {item}")
        else:
            st.info("Belum ada riwayat pencarian.")

elif page == "Rekomendasi (Novel)":
    st.title("Rekomendasi Berdasarkan Novel")

    novel_titles = sorted(novels_df['Title'].tolist())
    selected_novel_title = st.selectbox("Pilih Novel:", ['-- Pilih Novel --'] + novel_titles)

    if selected_novel_title and selected_novel_title != '-- Pilih Novel --':
        add_to_history(selected_novel_title)
        recommendations = get_recommendations_content_based(selected_novel_title, top_n=10)
        
        if not recommendations.empty:
            st.subheader(f"Rekomendasi untuk '{selected_novel_title}'")
            st.dataframe(recommendations[['Title', 'Genre', 'Score', 'Similarity_Score']].style.format({"Similarity_Score": "{:.4f}"}))
        else:
            st.warning(f"Novel '{selected_novel_title}' tidak ditemukan atau tidak ada rekomendasi.")

elif page == "Rekomendasi (Genre)":
    st.title("Rekomendasi Berdasarkan Genre")

    all_genres = set()
    for genre_list in novels_df['Genre'].dropna().unique():
        for g in genre_list.split(','):
            all_genres.add(g.strip())
    genres = sorted(list(all_genres))
    
    selected_genre = st.selectbox("Pilih Genre:", ['-- Pilih Genre --'] + genres)

    if selected_genre and selected_genre != '-- Pilih Genre --':
        add_to_history(selected_genre)
        genre_novels = novels_df[novels_df['Genre'].str.contains(selected_genre, case=False, na=False)].sort_values(by='Score', ascending=False)
        recommendations = genre_novels.head(10)
        
        if not recommendations.empty:
            st.subheader(f"Rekomendasi Novel Genre '{selected_genre}'")
            st.dataframe(recommendations[['Title', 'Genre', 'Score']])
        else:
            st.warning(f"Tidak ada novel dengan genre '{selected_genre}' atau tidak ada rekomendasi.")
