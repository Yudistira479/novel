import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# ---------------- CONFIG ----------------
st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# ---------------- LOAD DATA FROM ZIP ----------------
@st.cache_data
def load_data_from_zip(zip_path='novel_recommender_app.zip', filename='novels.csv'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(filename)
    df = pd.read_csv(filename)
    df = df.dropna(subset=['title', 'genre', 'rating'])  # pastikan kolom penting tidak kosong
    return df

df = load_data_from_zip()

# ---------------- PREPROCESS TF-IDF (Genre only) ----------------
@st.cache_data
def preprocess(df):
    df = df.copy()
    df['genre'] = df['genre'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])

    return df, tfidf_matrix, tfidf

df, tfidf_matrix, tfidf = preprocess(df)

# ---------------- TRAIN RANDOM FOREST REGRESSOR (Rating) ----------------
@st.cache_data
def train_random_forest(df):
    df_model = df.copy()
    features = ['views', 'likes', 'chapter_count', 'popularity', 'score']
    
    le = LabelEncoder()
    df_model['status_encoded'] = le.fit_transform(df_model['status'].astype(str))
    features.append('status_encoded')

    df_model = df_model.dropna(subset=features + ['rating'])
    X = df_model[features]
    y = df_model['rating']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le

rf_model, le_status = train_random_forest(df)

# ---------------- CLASSIFIER FOR RATING ----------------
@st.cache_data
def train_rf_classifier_rating(df):
    df_clf = df.copy()
    df_clf['rating_class'] = pd.cut(df_clf['rating'], bins=[0, 2, 4, 5], labels=['low', 'medium', 'high'])

    le = LabelEncoder()
    df_clf['status_encoded'] = le.fit_transform(df_clf['status'].astype(str))

    features = ['views', 'likes', 'chapter_count', 'popularity', 'score', 'status_encoded']
    df_clf = df_clf.dropna(subset=features + ['rating_class'])

    X = df_clf[features]
    y = df_clf['rating_class']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred, labels=['low', 'medium', 'high'])

    return model, le, cm, y, y_pred

# ---------------- CLASSIFIER FOR SCORE ----------------
@st.cache_data
def train_rf_classifier_score(df):
    df_clf = df.copy()
    df_clf['score_class'] = pd.cut(df_clf['score'], bins=[0, 2, 4, 6, 10], labels=['very_low', 'low', 'medium', 'high'])

    le = LabelEncoder()
    df_clf['status_encoded'] = le.fit_transform(df_clf['status'].astype(str))

    features = ['views', 'likes', 'chapter_count', 'popularity', 'status_encoded']
    df_clf = df_clf.dropna(subset=features + ['score_class'])

    X = df_clf[features]
    y = df_clf['score_class']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred, labels=['very_low', 'low', 'medium', 'high'])

    return model, le, cm, y, y_pred

# ---------------- SIDEBAR ----------------
st.sidebar.title("📚 Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Rekomendasi Rating (RF)", "Rekomendasi Genre (Content-Based)", "Evaluasi Model"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("📖 Daftar 10 Novel Terpopuler")
    top_novels = df.sort_values(by='popularity', ascending=False).head(10)
    st.dataframe(top_novels[['title', 'author', 'genre', 'rating', 'score', 'views', 'popularity']])

# ---------------- REKOMENDASI BERDASARKAN RATING ----------------
elif page == "Rekomendasi Rating (RF)":
    st.title("🎯 Rekomendasi Berdasarkan Rating (Random Forest)")
    selected_title = st.selectbox("Pilih Judul Novel", df['title'].unique())

    selected_novel = df[df['title'] == selected_title].iloc[0]

    input_features = pd.DataFrame([{
        'views': selected_novel['views'],
        'likes': selected_novel['likes'],
        'chapter_count': selected_novel['chapter_count'],
        'popularity': selected_novel['popularity'],
        'score': selected_novel['score'],
        'status_encoded': le_status.transform([selected_novel['status']])[0] if pd.notnull(selected_novel['status']) else 0
    }])

    predicted_rating = rf_model.predict(input_features)[0]
    st.write(f"📌 Prediksi Rating: **{predicted_rating:.2f}**")

    df_temp = df.copy()
    df_temp['status_encoded'] = le_status.transform(df_temp['status'].astype(str))
    df_temp = df_temp.dropna(subset=['score'])
    df_temp['predicted_rating'] = rf_model.predict(df_temp[['views', 'likes', 'chapter_count', 'popularity', 'score', 'status_encoded']])
    recommended = df_temp[df_temp['title'] != selected_title].copy()
    recommended['score_diff'] = abs(recommended['predicted_rating'] - predicted_rating)
    result = recommended.sort_values(by='score_diff').head(10)

    st.subheader("📚 Rekomendasi Novel dengan Rating Mirip")
    st.dataframe(result[['title', 'author', 'genre', 'rating', 'predicted_rating', 'score']])

# ---------------- CONTENT-BASED FILTERING ----------------
elif page == "Rekomendasi Genre (Content-Based)":
    st.title("🔍 Rekomendasi Berdasarkan Genre (TF-IDF)")
    selected_title = st.selectbox("Pilih Judul Novel", df['title'].unique(), key="genre")

    if selected_title:
        idx = df[df['title'] == selected_title].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        indices = cosine_sim.argsort()[-11:-1][::-1]

        st.subheader("📚 Rekomendasi Novel dengan Genre Mirip")
        st.dataframe(df.iloc[indices][['title', 'author', 'genre', 'rating', 'score']])

        st.subheader("🔍 Fitur TF-IDF dari Genre (Top 10 Term)")
        vector_index = tfidf_matrix[idx].nonzero()[1]
        tfidf_vals = tfidf_matrix[idx, vector_index].toarray().flatten()
        top_terms = np.array(tfidf.get_feature_names_out())[vector_index]

        tfidf_table = pd.DataFrame({
            'term': top_terms,
            'weight': tfidf_vals
        }).sort_values(by='weight', ascending=False).head(10)

        st.dataframe(tfidf_table)

# ---------------- EVALUASI MODEL ----------------
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model Random Forest")

    st.subheader("📊 Confusion Matrix Berdasarkan Rating")
    clf_rating, _, cm_rating, y_rating, y_pred_rating = train_rf_classifier_rating(df)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm_rating, annot=True, fmt='d', xticklabels=['low', 'medium', 'high'],
                yticklabels=['low', 'medium', 'high'], cmap='Blues', ax=ax1)
    st.pyplot(fig1)

    st.subheader("📊 Confusion Matrix Berdasarkan Score")
    clf_score, _, cm_score, y_score, y_pred_score = train_rf_classifier_score(df)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm_score, annot=True, fmt='d', xticklabels=['very_low', 'low', 'medium', 'high'],
                yticklabels=['very_low', 'low', 'medium', 'high'], cmap='Greens', ax=ax2)
    st.pyplot(fig2)
