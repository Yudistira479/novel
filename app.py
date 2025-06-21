import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

# === Fungsi Load Data dari GitHub ===
@st.cache_data
def load_data():
    url = "https://www.kaggle.com/code/fatahillahmuhammad/rekomendasi-novel/input?select=novels.csv"
    df = pd.read_csv(url)
    return df.dropna(subset=['title', 'genres', 'rating'])

# === Fungsi TF-IDF untuk Genre ===
def compute_genre_similarity(df):
    tfidf = TfidfVectorizer()
    genre_tfidf = tfidf.fit_transform(df['genres'])
    similarity_matrix = cosine_similarity(genre_tfidf)
    return similarity_matrix

# === Fungsi Random Forest untuk Rating Rekomendasi ===
def train_random_forest(df):
    features = df[['views', 'likes', 'chapter_count']]
    target = df['rating']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    return model

# === Fungsi Rekomendasi Berdasarkan Genre ===
def recommend_by_genre(df, similarity_matrix, title, top_n=5):
    if title not in df['title'].values:
        return []
    index = df[df['title'] == title].index[0]
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = [df.iloc[i[0]] for i in scores[1:top_n+1]]
    return recommendations

# === Fungsi Rekomendasi Berdasarkan Rating ===
def recommend_by_rating(df, title, model, top_n=5):
    if title not in df['title'].values:
        return []
    input_data = df[df['title'] == title][['views', 'likes', 'chapter_count']]
    predicted_rating = model.predict(input_data)[0]
    similar = df[(df['rating'] >= predicted_rating - 0.2) & (df['rating'] <= predicted_rating + 0.2)]
    return similar.head(top_n).to_dict('records')

# === Halaman Streamlit ===
st.set_page_config(page_title="📚 Rekomendasi Novel", layout="wide")

df = load_data()
similarity_matrix = compute_genre_similarity(df)
model = train_random_forest(df)

menu = st.sidebar.selectbox("Pilih Halaman", ["📌 Home", "📖 Rekomendasi Berdasarkan Judul"])

if menu == "📌 Home":
    st.title("📚 10 Novel Terpopuler")
    top10 = df.sort_values(by='popularity', ascending=False).head(10)
    st.dataframe(top10[['title', 'author', 'genres', 'rating', 'popularity']])

elif menu == "📖 Rekomendasi Berdasarkan Judul":
    st.title("🔍 Rekomendasi Novel Berdasarkan Judul")
    selected_title = st.selectbox("Pilih Judul Novel", df['title'].unique())

    if st.button("Rekomendasikan!"):
        genre_recs = recommend_by_genre(df, similarity_matrix, selected_title)
        rating_recs = recommend_by_rating(df, selected_title, model)

        st.subheader("📘 Rekomendasi Berdasarkan Genre:")
        for rec in genre_recs:
            st.markdown(f"**{rec['title']}** - {rec['genres']} (⭐ {rec['rating']})")

        st.subheader("⭐ Rekomendasi Berdasarkan Rating:")
        for rec in rating_recs:
            st.markdown(f"**{rec['title']}** - Rating: {rec['rating']}")
