import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set page configuration
st.set_page_config(layout="wide", page_title="Novel Recommendation App")

# --- Fungsi untuk memuat data dan model ---
@st.cache_data
def load_data():
    try:
        # Menggunakan raw link dari GitHub untuk novels.csv
        data_url = "https://raw.githubusercontent.com/Yudistira479/novel/main/novels.csv"
        df = pd.read_csv(data_url)

        # Mengisi nilai NaN di kolom 'genres', 'description', 'author'
        # Penting untuk mengisi NaN sebelum menggabungkan teks
        df['genres'] = df['genres'].fillna('')
        df['description'] = df['description'].fillna('')
        df['author'] = df['author'].fillna('Unknown')
        df['title'] = df['title'].fillna('Untitled')
        df['status'] = df['status'].fillna('Unknown')
        df['volume'] = df['volume'].fillna(0)
        df['favorites'] = df['favorites'].fillna(0)
        df['score'] = df['score'].fillna(0.0)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame() # Mengembalikan DataFrame kosong jika ada kesalahan

@st.cache_resource
def load_model_and_vectorizer(df):
    # Menggabungkan teks dari kolom 'description', 'genre', dan 'author' secara langsung
    # untuk TF-IDF, tanpa membuat kolom 'combined_features' baru di DataFrame.
    # Pastikan kolom-kolom ini telah diisi NaN sebelumnya.
    texts_for_tfidf = df['description'] + ' ' + df['genres'] + ' ' + df['author']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts_for_tfidf)

    # Menggunakan RandomForestRegressor untuk pemodelan
    # Ini adalah contoh, Anda mungkin perlu melatih model lebih lanjut
    # Untuk tujuan rekomendasi, kita akan lebih banyak menggunakan kesamaan TF-IDF
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Misalkan kita ingin memprediksi 'score' berdasarkan fitur teks (ini bisa lebih kompleks)
    # Untuk rekomendasi konten, kita lebih fokus pada kesamaan fitur
    # Di sini, kita hanya akan 'melatih' model dengan data dummy atau placeholder jika tidak ada target eksplisit
    # Karena ini content-based, kita akan menggunakan cosine similarity dari TF-IDF
    
    return tfidf, tfidf_matrix, model

df = load_data()

# Hanya lanjutkan jika DataFrame tidak kosong
if not df.empty:
    tfidf, tfidf_matrix, model = load_model_and_vectorizer(df)

    # Inisialisasi riwayat pencarian dalam session state
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

    # --- Sidebar untuk Navigasi ---
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Rekomendasi Berdasarkan Skor", "Rekomendasi Berdasarkan Genre", "Distribusi Data"])

    # --- Halaman 1: Beranda ---
    if page == "Beranda":
        st.title("Beranda: Rekomendasi Novel")
        st.write("Selamat datang di aplikasi rekomendasi novel!")

        st.header("10 Novel Terpopuler (Berdasarkan Favorit)")
        # Menampilkan 10 novel terpopuler berdasarkan kolom 'favorites'
        top_10_novels = df.sort_values(by='favorites', ascending=False).head(10)
        st.dataframe(top_10_novels[['title', 'author', 'genre', 'favorites', 'score']])

        st.header("Cari Novel dan Dapatkan Rekomendasi")
        novel_title_input = st.text_input("Masukkan judul novel:")

        if novel_title_input:
            # Tambahkan ke riwayat pencarian
            if novel_title_input not in st.session_state.search_history:
                st.session_state.search_history.append(novel_title_input)
            
            # Cari novel yang cocok
            matched_novels = df[df['title'].str.contains(novel_title_input, case=False, na=False)]

            if not matched_novels.empty:
                st.subheader(f"Hasil pencarian untuk '{novel_title_input}':")
                st.dataframe(matched_novels[['title', 'author', 'genre', 'score']])

                # Pilih novel untuk rekomendasi
                selected_novel_index = st.selectbox("Pilih novel dari hasil pencarian untuk mendapatkan rekomendasi:", matched_novels.index, format_func=lambda x: df.loc[x, 'title'])
                selected_novel = df.loc[selected_novel_index]

                st.subheader(f"Rekomendasi serupa untuk: {selected_novel['title']}")
                
                # Hitung cosine similarity
                idx = df[df['title'] == selected_novel['title']].index[0]
                
                # Dapatkan teks yang digunakan untuk TF-IDF untuk novel yang dipilih
                selected_novel_text = df.loc[idx, 'description'] + ' ' + df.loc[idx, 'genre'] + ' ' + df.loc[idx, 'author']
                
                # Transformasi teks novel yang dipilih menggunakan TF-IDF vectorizer yang sudah dilatih
                selected_novel_tfidf = tfidf.transform([selected_novel_text])

                # Hitung kesamaan kosinus antara novel yang dipilih dan semua novel lainnya
                cosine_similarities = linear_kernel(selected_novel_tfidf, tfidf_matrix).flatten()
                
                # Dapatkan indeks novel serupa, kecuali novel itu sendiri
                related_novel_indices = cosine_similarities.argsort()[:-12:-1] # Top 10 + itself
                recommended_novels = df.iloc[related_novel_indices].drop(selected_novel_index, errors='ignore') # Remove the selected novel itself

                st.dataframe(recommended_novels[['title', 'author', 'genre', 'score', 'favorites']].head(10))
            else:
                st.warning(f"Tidak ada novel yang ditemukan dengan judul '{novel_title_input}'.")

        st.header("Riwayat Pencarian")
        if st.session_state.search_history:
            for i, history_item in enumerate(st.session_state.search_history):
                st.write(f"{i+1}. {history_item}")
            if st.button("Bersihkan Riwayat Pencarian"):
                st.session_state.search_history = []
                st.experimental_rerun()
        else:
            st.info("Riwayat pencarian kosong.")


    # --- Halaman 2: Rekomendasi Berdasarkan Skor ---
    elif page == "Rekomendasi Berdasarkan Skor":
        st.title("Rekomendasi Berdasarkan Skor")
        st.write("Temukan novel dengan skor tertinggi!")

        num_recommendations_score = st.slider("Jumlah novel untuk direkomendasikan:", 5, 50, 10)
        
        # Urutkan berdasarkan skor dan tampilkan
        top_score_novels = df.sort_values(by='score', ascending=False).head(num_recommendations_score)
        st.dataframe(top_score_novels[['title', 'author', 'genre', 'score', 'favorites']])

    # --- Halaman 3: Rekomendasi Berdasarkan Genre ---
    elif page == "Rekomendasi Berdasarkan Genre":
        st.title("Rekomendasi Berdasarkan Genre")
        st.write("Temukan novel berdasarkan genre favorit Anda.")

        all_genres = sorted(list(set(g for genres_str in df['genre'].dropna() for g in genres_str.split(', '))))
        selected_genre = st.selectbox("Pilih Genre:", ['Semua Genre'] + all_genres)

        if selected_genre == 'Semua Genre':
            filtered_novels = df
        else:
            filtered_novels = df[df['genre'].str.contains(selected_genre, case=False, na=False)]
        
        num_recommendations_genre = st.slider("Jumlah novel untuk direkomendasikan:", 5, 50, 10)
        
        if not filtered_novels.empty:
            # Urutkan berdasarkan favorit atau skor untuk rekomendasi yang lebih baik dalam genre
            recommended_by_genre = filtered_novels.sort_values(by='favorites', ascending=False).head(num_recommendations_genre)
            st.dataframe(recommended_by_genre[['title', 'author', 'genre', 'score', 'favorites']])
        else:
            st.warning(f"Tidak ada novel yang ditemukan untuk genre '{selected_genre}'.")

    # --- Halaman 4: Distribusi Data ---
    elif page == "Distribusi Data":
        st.title("Distribusi Data Novel")
        st.write("Visualisasi distribusi fitur-fitur novel.")

        st.header("Distribusi Genre")
        # Membersihkan dan menghitung frekuensi genre
        genres_list = df['genre'].dropna().apply(lambda x: [g.strip() for g in x.split(', ')])
        all_genres_flat = [item for sublist in genres_list for item in sublist]
        genre_counts = pd.Series(all_genres_flat).value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Count']
        
        fig_genre = px.bar(genre_counts.head(20), x='Genre', y='Count', title='Top 20 Distribusi Genre Novel')
        st.plotly_chart(fig_genre, use_container_width=True)

        st.header("Distribusi Status")
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_status = px.pie(status_counts, values='Count', names='Status', title='Distribusi Status Novel')
        st.plotly_chart(fig_status, use_container_width=True)

        st.header("Distribusi Volume")
        # Mengelompokkan volume agar visualisasi tidak terlalu padat
        df['volume_bin'] = pd.cut(df['volume'], bins=5, labels=False, include_lowest=True)
        volume_bins_labels = [f"Bin {i}" for i in sorted(df['volume_bin'].unique())]
        volume_counts = df['volume_bin'].value_counts().sort_index().reset_index()
        volume_counts.columns = ['Volume Bin Index', 'Count']
        
        # Membuat label yang lebih deskriptif
        volume_counts['Volume Range'] = volume_counts['Volume Bin Index'].apply(lambda x: f"{df['volume'].min() + x * (df['volume'].max() - df['volume'].min()) / 5:.0f} - {df['volume'].min() + (x+1) * (df['volume'].max() - df['volume'].min()) / 5:.0f}")

        fig_volume = px.bar(volume_counts, x='Volume Range', y='Count', title='Distribusi Volume Novel (Binned)')
        st.plotly_chart(fig_volume, use_container_width=True)

        st.header("Distribusi Favorit")
        # Mengelompokkan favorit
        df['favorites_bin'] = pd.cut(df['favorites'], bins=10, labels=False, include_lowest=True)
        favorites_counts = df['favorites_bin'].value_counts().sort_index().reset_index()
        favorites_counts.columns = ['Favorites Bin Index', 'Count']
        
        # Membuat label yang lebih deskriptif
        favorites_counts['Favorites Range'] = favorites_counts['Favorites Bin Index'].apply(lambda x: f"{df['favorites'].min() + x * (df['favorites'].max() - df['favorites'].min()) / 10:.0f} - {df['favorites'].min() + (x+1) * (df['favorites'].max() - df['favorites'].min()) / 10:.0f}")

        fig_favorites = px.bar(favorites_counts, x='Favorites Range', y='Count', title='Distribusi Favorit Novel (Binned)')
        st.plotly_chart(fig_favorites, use_container_width=True)

        st.header("Distribusi Skor")
        fig_score = px.histogram(df, x='score', nbins=20, title='Distribusi Skor Novel', labels={'score': 'Skor'})
        st.plotly_chart(fig_score, use_container_width=True)

else:
    st.error("Gagal memuat data. Mohon periksa URL GitHub atau format file.")
