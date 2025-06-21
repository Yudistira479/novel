import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
DATA_PATH = 'novels.csv'

# Muat dataset novel
try:
    novels_df = pd.read_csv(DATA_PATH)
    novels_df.fillna('', inplace=True) # Mengisi nilai NaN dengan string kosong
    # Pastikan kolom yang dibutuhkan ada
    required_columns = ['Title', 'Description', 'Genre', 'Status', 'Volume', 'Favorites', 'Views', 'Score', 'Tags']
    for col in required_columns:
        if col not in novels_df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam novels.csv. Pastikan nama kolom sesuai.")
except FileNotFoundError:
    print(f"Error: File '{DATA_PATH}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan app.py.")
    exit()
except Exception as e:
    print(f"Error saat memuat atau memproses novels.csv: {e}")
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

@app.route('/data_distribution')
def data_distribution():
    """Halaman untuk menampilkan distribusi data."""
    plots = {}

    # Distribusi Genre
    plt.figure(figsize=(12, 6))
    # Explode the 'Genre' column and count occurrences for more accurate distribution
    all_genres_exploded = novels_df['Genre'].dropna().str.split(',').explode()
    genre_counts = all_genres_exploded.str.strip().value_counts().head(15) # Ambil 15 genre teratas
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
    plt.title('Distribusi Genre Novel (Top 15)')
    plt.xlabel('Genre')
    plt.ylabel('Jumlah Novel')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    plots['genre'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Distribusi Status
    plt.figure(figsize=(8, 5))
    status_counts = novels_df['Status'].value_counts()
    sns.barplot(x=status_counts.index, y=status_counts.values, palette='magma')
    plt.title('Distribusi Status Novel')
    plt.xlabel('Status')
    plt.ylabel('Jumlah Novel')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    plots['status'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Distribusi Volume (jika ada kolom Volume numerik)
    novels_df['Volume'] = pd.to_numeric(novels_df['Volume'], errors='coerce')
    if 'Volume' in novels_df.columns and novels_df['Volume'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.histplot(novels_df['Volume'].dropna(), bins=20, kde=True, color='purple')
        plt.title('Distribusi Volume Novel')
        plt.xlabel('Volume')
        plt.ylabel('Jumlah Novel')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        plots['volume'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        plots['volume'] = None # Tidak membuat plot jika Volume bukan numerik

    # Distribusi Favorites
    novels_df['Favorites'] = pd.to_numeric(novels_df['Favorites'], errors='coerce')
    if 'Favorites' in novels_df.columns and novels_df['Favorites'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.histplot(novels_df['Favorites'].dropna(), bins=20, kde=True, color='teal')
        plt.title('Distribusi Jumlah Favorites')
        plt.xlabel('Jumlah Favorites')
        plt.ylabel('Jumlah Novel')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        plots['favorites'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        plots['favorites'] = None # Tidak membuat plot jika Favorites bukan numerik

    return render_template('data_distribution.html', plots=plots)

@app.route('/confusion_matrix')
def show_confusion_matrix():
    """Halaman untuk menampilkan Confusion Matrix dan akurasi."""
    try:
        # Load data pelatihan (ini harusnya dari model_pelatihan.py)
        # Untuk demo ini, kita akan membuat prediksi dummy jika tidak ada data asli
        # Di aplikasi nyata, Anda akan memiliki set data uji yang terpisah
        
        # Menggunakan kolom 'Genre' sebagai target
        novels_df['target_genre'] = novels_df['Genre'].apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')
        
        # Hapus baris dengan 'Unknown' genre jika ini tidak diinginkan
        df_for_cm = novels_df[novels_df['target_genre'] != 'Unknown'].copy()

        if df_for_cm.empty:
            cm_plot = None
            accuracy = "Tidak ada data yang valid untuk membuat Confusion Matrix setelah preprocessing genre."
            print("Tidak ada data yang valid untuk membuat Confusion Matrix setelah preprocessing genre.")
            return render_template('confusion_matrix.html', cm_plot=cm_plot, accuracy=accuracy)

        # Pastikan tfidf_matrix sudah diinisialisasi
        # Untuk tujuan ini, kita perlu tfidf_vectorizer untuk mentransformasi data yang digunakan untuk CM
        X_sample = tfidf_vectorizer.transform(df_for_cm['combined_features'])
        y_true_sample = df_for_cm['target_genre']

        # Pastikan model sudah dilatih dan bisa memprediksi
        if model:
            y_pred_sample = model.predict(X_sample)
            
            # Filter hanya label yang ada di y_true_sample dan y_pred_sample
            unique_labels = np.unique(np.concatenate((y_true_sample, y_pred_sample)))
            
            # Hitung akurasi
            accuracy = np.mean(y_pred_sample == y_true_sample) * 100
            
            # Buat Confusion Matrix
            cm = confusion_matrix(y_true_sample, y_pred_sample, labels=unique_labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
            plt.title('Confusion Matrix Prediksi Genre')
            plt.xlabel('Prediksi')
            plt.ylabel('Aktual')
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            cm_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
        else:
            cm_plot = None
            accuracy = "Model belum dilatih atau dimuat."

    except Exception as e:
        cm_plot = None
        accuracy = f"Error saat membuat Confusion Matrix: {e}"
        print(f"Error saat membuat Confusion Matrix: {e}")

    return render_template('confusion_matrix.html', cm_plot=cm_plot, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
