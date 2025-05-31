import streamlit as st
st.set_page_config(page_title="Novel Recommendation App", layout="wide")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('novels.csv')
    return df

df = load_data()

# Sidebar navigation
page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Rekomendasi Scored", "Rekomendasi Genre"])

# Halaman Home
if page == "Home":
    st.title("📚 Daftar Novel Populer")
    top_novels = df.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_novels[['title', 'authors', 'genres', 'scored', 'popularty']])

    st.markdown("---")
    st.subheader("Riwayat Rekomendasi")
    st.write("Belum ada riwayat rekomendasi.")

# Halaman Rekomendasi Berdasarkan Scored
elif page == "Rekomendasi Scored":
    st.title("🔍 Rekomendasi Berdasarkan Scored")

    title_input = st.selectbox("Pilih Judul Novel", df['title'].values)

    selected_novel = df[df['title'] == title_input].iloc[0]
    X = df[['scored']]
    y = df['popularty']

    model = RandomForestRegressor()
    model.fit(X, y)
    pred = model.predict(X)

    df['predicted'] = pred
    df['scored_diff'] = abs(df['scored'] - selected_novel['scored'])
    recommended = df[df['title'] != title_input].sort_values(by='scored_diff').head(5)

    st.subheader(f"Rekomendasi untuk: {title_input}")
    st.dataframe(recommended[['title', 'authors', 'genres', 'scored']])

# Halaman Rekomendasi Berdasarkan Genre
elif page == "Rekomendasi Genre":
    st.title("📖 Rekomendasi Berdasarkan Genre")

    title_input = st.selectbox("Pilih Judul Novel", df['title'].values, key="genre")
    selected_novel = df[df['title'] == title_input].iloc[0]

    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['genres'])

    X = df[['genre_encoded']]
    y = df['title']

    model = RandomForestClassifier()
    model.fit(X, y)

    genre_code = le.transform([selected_novel['genres']])[0]
    df['genre_diff'] = abs(df['genre_encoded'] - genre_code)

    recommended = df[df['title'] != title_input].sort_values(by='genre_diff').head(5)

    st.subheader(f"Rekomendasi untuk genre: {selected_novel['genres']}")
    st.dataframe(recommended[['title', 'authors', 'genres', 'scored']])
