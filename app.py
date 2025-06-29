import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------ Load Data ------------------
df = pd.read_csv('novels_selected.csv')

# ------------------ Ekstraksi Fitur TF-IDF ------------------
df['title'] = df['title'].fillna('').str.lower().str.replace('[^a-zA-Z]', ' ', regex=True).str.replace('\s+', ' ', regex=True).str.strip()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])

# ------------------ Rekomendasi Berdasarkan Scored ------------------
def rekomendasi_berdasarkan_score(input_score):
    if not isinstance(input_score, (int, float)):
        print("❌ Input skor tidak valid. Harus berupa angka.")
        return None

    X = df[['score']]
    y = df['popularty']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    predicted_pop = model.predict([[input_score]])[0]
    df['predicted_popularty'] = model.predict(df[['score']])
    df['predicted_diff'] = abs(df['predicted_popularty'] - predicted_pop)

    tfidf_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    input_idx = np.argmin(abs(df['score'] - input_score))
    tfidf_scores = tfidf_similarities[input_idx]
    df['tfidf_sim'] = tfidf_scores

    df['combined_score'] = df['tfidf_sim'] - df['predicted_diff'] / df['predicted_diff'].max()
    recommended = df.sort_values(by='combined_score', ascending=False).head(5)
    return recommended[['title', 'author', 'type','genre', 'score', 'popularty', 'predicted_popularty']]

# ------------------ Rekomendasi Berdasarkan Genre & Judul Serupa ------------------
def rekomendasi_berdasarkan_genre(title_input):
    if not title_input.strip():
        print("❌ Judul tidak boleh kosong.")
        return None

    matched_titles = df[df['title'].str.contains(title_input, case=False, na=False)]
    if matched_titles.empty:
        return None

    selected_genre = matched_titles.iloc[0]['genre']
    genre_novels = df[df['genre'] == selected_genre].copy()
    X_genre = genre_novels[['score']]
    y_genre = genre_novels['popularty']
    model_genre = RandomForestRegressor(n_estimators=100, random_state=42)
    model_genre.fit(X_genre, y_genre)

    genre_novels['predicted_popularty'] = model_genre.predict(X_genre)

    genre_indices = genre_novels.index
    title_vector = tfidf_vectorizer.transform([title_input])
    genre_tfidf = tfidf_matrix[genre_indices]
    tfidf_sim = cosine_similarity(title_vector, genre_tfidf).flatten()
    genre_novels['tfidf_sim'] = tfidf_sim

    genre_novels['combined_score'] = genre_novels['tfidf_sim'] + genre_novels['predicted_popularty'] / genre_novels['predicted_popularty'].max()

    recommended = genre_novels.sort_values(by='combined_score', ascending=False).head(5)
    return recommended[['title', 'author', 'type','genre', 'score', 'popularty', 'predicted_popularty']]

# ------------------ Visualisasi Distribusi ------------------
def visualisasi_distribusi():
    plt.figure(figsize=(10, 6))
    top_genres = df['genre'].value_counts().head(10)
    top_genres.plot(kind='bar', color='skyblue')
    plt.title('Distribusi 10 Genre Terpopuler')
    plt.xlabel('Genre')
    plt.ylabel('Jumlah Novel')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    status_counts = df['status'].value_counts()
    status_counts.plot(kind='bar', color='salmon')
    plt.title('Distribusi Status Novel')
    plt.xlabel('Status')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    if 'years start' in df.columns:
        plt.figure(figsize=(10, 6))
        df['years start'].dropna().astype(int).value_counts().sort_index().plot(kind='bar', color='lightgreen')
        plt.title('Distribusi Tahun Terbit Novel')
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Novel')
        plt.tight_layout()
        plt.show()

# ------------------ Contoh Penggunaan ------------------
if __name__ == "__main__":
    print("Rekomendasi berdasarkan skor:")
    try:
        skor_input = float(input("Masukkan skor: "))
        hasil_skor = rekomendasi_berdasarkan_score(skor_input)
        if hasil_skor is not None:
            print(hasil_skor.to_string(index=False))
    except ValueError:
        print("❌ Input skor tidak valid. Harus berupa angka.")

    print("\nRekomendasi berdasarkan genre dan judul:")
    judul_input = input("Masukkan judul: ").strip()
    hasil_genre = rekomendasi_berdasarkan_genre(judul_input)
    if hasil_genre is not None:
        print(hasil_genre.to_string(index=False))
    else:
        print("❌ Judul tidak ditemukan atau tidak valid.")

    print("\nVisualisasi distribusi genre, status, dan tahun terbit:")
    visualisasi_distribusi()
