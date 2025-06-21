import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------- SETUP ----------
st.set_page_config(page_title="📚 Novel Recommender", layout="wide")
st.title("📚 Novel Recommendation System")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Yudistira479/novel/main/novels_cleaned%20(1).csv"
    df = pd.read_csv(url, delimiter=";")
    df = df.dropna(subset=['title', 'genre', 'score'])
    return df

df = load_data()

# ---------- TF-IDF FEATURE EXTRACTION ----------
@st.cache_data
def tfidf_features(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    return tfidf_matrix

tfidf_matrix = tfidf_features(df)

# ---------- RANDOM FOREST CLASSIFIER FOR SCORE ----------
@st.cache_data
def train_rf_model(df):
    df = df.copy()
    df = df.dropna(subset=['views', 'likes', 'chapter_count', 'popularity', 'status', 'score'])
    df['score_class'] = pd.cut(df['score'], bins=[0, 2, 4, 6, 10], labels=['very_low', 'low', 'medium', 'high'])
    le = LabelEncoder()
    df['status_encoded'] = le.fit_transform(df['status'].astype(str))
    features = ['views', 'likes', 'chapter_count', 'popularity', 'status_encoded']
    X = df[features]
    y = df['score_class']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    return model, le, cm, model.classes_

rf_model, le_status, confusion_mtx, score_classes = train_rf_model(df)

# ---------- SIDEBAR NAVIGATION ----------
page = st.sidebar.radio("📂 Pilih Halaman", ["Home", "Rekomendasi Score", "Rekomendasi Genre", "Distribusi Data"])

# ---------- HISTORY STORAGE ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- PAGE 1: HOME ----------
if page == "Home":
    st.header("📖 10 Novel Terpopuler")
    top10 = df.sort_values(by="popularity", ascending=False).head(10)
    st.dataframe(top10[['title', 'author', 'genre', 'score', 'popularity']])
    
    st.subheader("🕓 Riwayat Pencarian")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history, columns=["Judul Novel"]))

# ---------- PAGE 2: REKOMENDASI SCORE ----------
elif page == "Rekomendasi Score":
    st.header("🎯 Rekomendasi Berdasarkan Score")
    selected = st.selectbox("Pilih Judul Novel", df["title"].unique())
    selected_novel = df[df["title"] == selected].iloc[0]

    if selected not in st.session_state.history:
        st.session_state.history.append(selected)

    # Preprocess input for model
    status_enc = le_status.transform([selected_novel["status"]])[0]
    input_features = pd.DataFrame([{
        "views": selected_novel["views"],
        "likes": selected_novel["likes"],
        "chapter_count": selected_novel["chapter_count"],
        "popularity": selected_novel["popularity"],
        "status_encoded": status_enc
    }])
    predicted_score_class = rf_model.predict(input_features)[0]
    st.markdown(f"🔮 **Prediksi Score Class: {predicted_score_class}**")

    # Rekomendasi novel dengan kelas score sama
    same_class = df.copy()
    same_class['status_encoded'] = le_status.transform(same_class['status'].astype(str))
    same_class['score_class'] = pd.cut(same_class['score'], bins=[0, 2, 4, 6, 10], labels=['very_low', 'low', 'medium', 'high'])
    filtered = same_class[(same_class['score_class'] == predicted_score_class) & (same_class['title'] != selected)]
    st.subheader("📚 Rekomendasi dengan Score Serupa")
    st.dataframe(filtered[['title', 'author', 'genre', 'score']].head(10))

    # Confusion matrix
    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_mtx, annot=True, fmt='d', xticklabels=score_classes, yticklabels=score_classes, cmap='Oranges', ax=ax)
    st.pyplot(fig)

# ---------- PAGE 3: REKOMENDASI GENRE ----------
elif page == "Rekomendasi Genre":
    st.header("🔍 Rekomendasi Berdasarkan Genre")
    selected = st.selectbox("Pilih Judul Novel", df["title"].unique(), key="genre")
    if selected not in st.session_state.history:
        st.session_state.history.append(selected)

    idx = df[df["title"] == selected].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-11:-1][::-1]
    recommended = df.iloc[similar_indices]
    
    st.subheader("📚 Genre Serupa")
    st.dataframe(recommended[['title', 'author', 'genre', 'score']])

# ---------- PAGE 4: DISTRIBUSI ----------
elif page == "Distribusi Data":
    st.header("📊 Distribusi Data")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Genre (Top 10)")
        genre_counts = df['genre'].value_counts().head(10)
        fig1, ax1 = plt.subplots()
        genre_counts.plot(kind='barh', ax=ax1, color='skyblue')
        st.pyplot(fig1)

    with col2:
        st.subheader("Distribusi Status")
        status_counts = df['status'].value_counts()
        fig2, ax2 = plt.subplots()
        status_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel('')
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Distribusi Volume")
        fig3, ax3 = plt.subplots()
        sns.histplot(df['volume'].dropna(), bins=20, kde=True, ax=ax3)
        st.pyplot(fig3)

    with col4:
        st.subheader("Distribusi Favorites")
        fig4, ax4 = plt.subplots()
        sns.histplot(df['favorites'].dropna(), bins=20, kde=True, ax=ax4)
        st.pyplot(fig4)
