import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix # Meskipun dihapus fiturnya, import ini masih ada di model_pelatihan.py
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' # Ganti dengan kunci rahasia yang kuat

# Path untuk menyimpan model dan vectorizer
MODEL_PATH = 'model_pelatihan.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
# URL dataset novel dari GitHub
DATA_URL = 'https://raw.githubusercontent.com/Yudistira479/novel/main/novels.csv'

# Muat dataset novel dari URL GitHub
try:
    novels_df = pd.read_csv(DATA_URL)
    novels_df.fillna('', inplace=True) # Mengisi nilai NaN dengan string kosong
    # Pastikan kolom yang dibutuhkan ada
    required_columns = ['Title', 'Description', 'Genre', 'Status', 'Volume', 'Favorites', 'Views', 'Score', 'Tags']
    for col in required_columns:
        if col not in novels_df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset dari GitHub. Pastikan nama kolom sesuai.")
except Exception as e:
    print(f"Error saat memuat atau memproses novels.csv dari GitHub: {e}")
    exit()

# Cek apakah model dan vectorizer sudah ada, jika tidak, arahkan untuk melatihnya
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Model atau TF-IDF Vectorizer tidak ditemukan. Silakan jalankan model_pelatihan.py terlebih dahulu.")
    # Opsional: Bisa mengarahkan pengguna ke halaman error atau otomatis melatih
    # Namun, lebih baik dilakukan secara terpisah untuk performa produksi
    exit()

# Muat model dan vectorizer yang sudah dilatih
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error saat memuat model atau vectorizer: {e}")
    exit()

# Kolom yang akan digunakan untuk TF-IDF
tfidf_columns = ['Title', 'Description', 'Genre', 'Tags']
novels_df['combined_features'] = novels_df[tfidf_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
tfidf_matrix = tfidf_vectorizer.transform(novels_df['combined_features'])

# Inisialisasi riwayat pencarian
if 'search_history' not in session:
    session['search_history'] = []

def get_recommendations_content_based(novel_title, top_n=10):
    """
    Memberikan rekomendasi novel berdasarkan kemiripan konten.
    """
    if novel_title not in novels_df['Title'].values:
        return pd.DataFrame() # Mengembalikan DataFrame kosong jika novel tidak ditemukan

    idx = novels_df[novels_df['Title'] == novel_title].index[0]
    novel_features = tfidf_matrix[idx]

    # Hitung kemiripan kosinus antara novel yang dipilih dan semua novel lainnya
    # tfidf_matrix sudah dihitung di awal, jadi kita bisa langsung menggunakannya
    cosine_similarities = (tfidf_matrix * novel_features.T).toarray().flatten()
    
    # Dapatkan indeks novel yang paling mirip (termasuk novel itu sendiri)
    similar_indices = cosine_similarities.argsort()[::-1]
    
    # Filter novel yang sama dan ambil top_n
    similar_novels = novels_df.iloc[similar_indices].copy()
    similar_novels['Similarity_Score'] = cosine_similarities[similar_indices]
    
    similar_novels = similar_novels[similar_novels['Title'] != novel_title]
    
    return similar_novels.head(top_n)

def add_to_history(query):
    """Menambahkan query ke riwayat pencarian."""
    if query not in session['search_history']:
        session['search_history'].insert(0, query) # Tambahkan di awal
    if len(session['search_history']) > 5: # Batasi riwayat hingga 5 item
        session['search_history'].pop()

@app.route('/')
def home():
    """Halaman utama: Menampilkan 10 novel terpopuler dan riwayat pencarian."""
    # Pastikan 'Views' adalah numerik dan isi NaN dengan 0 untuk pengurutan
    novels_df['Views'] = pd.to_numeric(novels_df['Views'], errors='coerce').fillna(0)
    top_novels = novels_df.sort_values(by='Views', ascending=False).head(10)
    
    # Ambil riwayat pencarian dari session
    history = session.get('search_history', [])
    
    return render_template('home.html', top_novels=top_novels, search_history=history)

@app.route('/recommend_by_score', methods=['GET', 'POST'])
def recommend_by_score():
    """Halaman rekomendasi berdasarkan score."""
    recommendations = pd.DataFrame()
    selected_novel_title = None

    if request.method == 'POST':
        selected_novel_title = request.form.get('novel_title')
        if selected_novel_title:
            add_to_history(selected_novel_title)
            recommendations = get_recommendations_content_based(selected_novel_title, top_n=10)
            if recommendations.empty:
                message = f"Novel '{selected_novel_title}' tidak ditemukan atau tidak ada rekomendasi."
            else:
                message = None
        else:
            message = "Silakan pilih novel."
    else:
        message = "Pilih novel untuk mendapatkan rekomendasi berdasarkan novel yang dipilih."

    return render_template('recommend_by_score.html', 
                           novels=novels_df[['Title', 'Genre', 'Score']].sort_values(by='Title').to_dict('records'), 
                           recommendations=recommendations.to_dict('records'),
                           selected_novel=selected_novel_title,
                           message=message)

@app.route('/recommend_by_genre', methods=['GET', 'POST'])
def recommend_by_genre():
    """Halaman rekomendasi berdasarkan genre."""
    recommendations = pd.DataFrame()
    selected_genre = None

    if request.method == 'POST':
        selected_genre = request.form.get('genre')
        if selected_genre:
            add_to_history(selected_genre)
            # Filter novel berdasarkan genre yang dipilih
            genre_novels = novels_df[novels_df['Genre'].str.contains(selected_genre, case=False, na=False)].sort_values(by='Score', ascending=False)
            # Ambil 10 novel teratas dari genre tersebut (atau semua jika kurang dari 10)
            recommendations = genre_novels.head(10)
            if recommendations.empty:
                message = f"Tidak ada novel dengan genre '{selected_genre}' atau tidak ada rekomendasi."
            else:
                message = None
        else:
            message = "Silakan pilih genre."
    else:
        message = "Pilih genre untuk mendapatkan rekomendasi."

    # Mengambil daftar genre unik dari kolom 'Genre'
    # Asumsikan 'Genre' bisa berisi string tunggal atau string yang dipisahkan koma
    all_genres = set()
    for genre_list in novels_df['Genre'].dropna().unique():
        for g in genre_list.split(','):
            all_genres.add(g.strip())
    
    genres = sorted(list(all_genres)) # Hapus duplikat dan urutkan
    
    return render_template('recommend_by_genre.html', 
                           genres=genres, 
                           recommendations=recommendations.to_dict('records'),
                           selected_genre=selected_genre,
                           message=message)

# Halaman distribusi data dan confusion matrix telah dihapus dari aplikasi.
# Jika Anda ingin menambahkan kembali salah satunya di masa depan,
# Anda bisa mengembalikan kode yang relevan dan menambahkan link navigasi.

if __name__ == '__main__':
    app.run(debug=True)
