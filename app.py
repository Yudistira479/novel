import streamlit as st
st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    df = pd.read_csv('novels.csv')
    return df

df = load_data()

# Inisialisasi session state untuk menyimpan riwayat
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Home", "⭐ Rekomendasi Scored", "🎯 Rekomendasi Genre"])

# ---------------------- Home Page ----------------------
if page == "🏠 Home":
    st.title("📚 Daftar Novel Populer")
    st.markdown("Berikut adalah daftar **10 novel paling populer** berdasarkan data:")

    top_novels = df.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_novels[['title', 'authors', 'genres', 'scored', 'popularty']], use_container_width=True)

    st.markdown("---")
    st.subheader("📜 Riwayat Rekomendasi")
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            st.markdown(f"### 🔎 Rekomendasi berdasarkan novel: <span style='color:green'><code>{item['judul_dipilih']}</code></span>", unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi di sidebar.")

# ------------------ Rekomendasi Berdasarkan Scored ------------------
elif page == "⭐ Rekomendasi Scored":
    st.title("⭐ Rekomendasi Novel Berdasarkan Scored")
    st.markdown("Pilih sebuah judul novel dan sistem akan merekomendasikan novel lain dengan **scored serupa** menggunakan algoritma **Random Forest**.")

    title_input = st.selectbox("📖 Pilih Judul Novel", df['title'].values)
    selected_novel = df[df['title'] == title_input].iloc[0]

    X = df[['scored']]
    y = df['popularty']
    model = RandomForestRegressor()
    model.fit(X, y)
    df['scored_diff'] = abs(df['scored'] - selected_novel['scored'])
    recommended = df[df['title'] != title_input].sort_values(by='scored_diff').head(5)

    st.markdown(f"### 🔍 Rekomendasi berdasarkan novel: <span style='color:green'><code>{title_input}</code></span>", unsafe_allow_html=True)
    st.dataframe(recommended[['title', 'authors', 'genres', 'scored']], use_container_width=True)

    # Tambahkan ke riwayat dengan tabel
    st.session_state.history.append({
        'judul_dipilih': title_input,
        'metode': 'scored',
        'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
    })

# ------------------ Rekomendasi Berdasarkan Genre ------------------
elif page == "🎯 Rekomendasi Genre":
    st.title("🎯 Rekomendasi Novel Berdasarkan Genre")
    st.markdown("Pilih sebuah judul novel dan sistem akan memberikan rekomendasi novel dengan **genre serupa** menggunakan algoritma **Random Forest**.")

    title_input = st.selectbox("📖 Pilih Judul Novel", df['title'].values, key="genre")
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

    st.markdown(f"### 📌 Rekomendasi berdasarkan genre: <span style='color:green'><code>{selected_novel['genres']}</code></span>", unsafe_allow_html=True)
    st.dataframe(recommended[['title', 'authors', 'genres', 'scored']], use_container_width=True)

    # Tambahkan ke riwayat
    st.session_state.history.append({
        'judul_dipilih': title_input,
        'metode': 'genre',
        'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
    })
