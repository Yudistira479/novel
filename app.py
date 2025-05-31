import streamlit as st
st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
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

# CSS styling untuk mempercantik
st.markdown("""
<style>
h1, h2, h3, h4 {
    color: #2E8B57;
}
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}
.stButton>button {
    color: white;
    background-color: #2E8B57;
    border-radius: 10px;
}
.stTable {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Home", "⭐ Rekomendasi Scored", "🎯 Rekomendasi Genre", "📊 Distribusi Novel"])

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
            st.markdown(f"### 🔎 Rekomendasi berdasarkan: <span style='color:green'><code>{item['judul_dipilih']}</code></span>", unsafe_allow_html=True)
            st.table(item['rekomendasi'])
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi di sidebar.")

# ------------------ Rekomendasi Berdasarkan Scored ------------------
elif page == "⭐ Rekomendasi Scored":
    st.title("⭐ Rekomendasi Novel Berdasarkan Scored")
    st.markdown("Masukkan skor dan sistem akan merekomendasikan novel dengan **scored serupa** menggunakan algoritma **Random Forest**.")

    input_score = st.slider("🎯 Pilih Nilai Skor", min_value=float(df['scored'].min()), max_value=float(df['scored'].max()), value=float(df['scored'].mean()), step=0.01)

    X = df[['scored']]
    y = df['popularty']
    model = RandomForestRegressor()
    model.fit(X, y)

    df['scored_diff'] = abs(df['scored'] - input_score)
    recommended = df.sort_values(by='scored_diff').head(5)

    st.markdown(f"### 🔍 Rekomendasi berdasarkan skor: <span style='color:green'><code>{input_score:.2f}</code></span>", unsafe_allow_html=True)
    st.dataframe(recommended[['title', 'authors', 'genres', 'scored']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Scored {input_score:.2f}',
        'metode': 'scored',
        'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
    })

# ------------------ Rekomendasi Berdasarkan Genre ------------------
elif page == "🎯 Rekomendasi Genre":
    st.title("🎯 Rekomendasi Novel Berdasarkan Genre")
    st.markdown("Masukkan genre novel dan sistem akan merekomendasikan novel dengan genre yang sesuai menggunakan algoritma **Random Forest**.")

    genre_input = st.text_input("✏️ Masukkan Genre Novel (case-sensitive)")

    if genre_input:
        genre_filtered = df[df['genres'] == genre_input]

        if not genre_filtered.empty:
            X = pd.get_dummies(df['genres'])
            y = df['scored']
            model = RandomForestRegressor()
            model.fit(X, y)

            input_vector = pd.get_dummies(pd.Series([genre_input]))
            input_vector = input_vector.reindex(columns=X.columns, fill_value=0)

            df['genre_score'] = model.predict(X)
            recommended = df[df['genres'] == genre_input].sort_values(by='scored', ascending=False).head(5)

            st.markdown(f"### 📌 Rekomendasi berdasarkan genre: <span style='color:green'><code>{genre_input}</code></span>", unsafe_allow_html=True)
            st.dataframe(recommended[['title', 'authors', 'genres', 'scored']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': genre_input,
                'metode': 'genre',
                'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
            })
        else:
            st.warning("Genre tidak ditemukan dalam data.")

# ------------------ Distribusi Genre dan Status ------------------
elif page == "📊 Distribusi Novel":
    st.title("📊 Distribusi Genre dan Status Novel")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎭 Distribusi Genre")
        genre_counts = df['genres'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        if 'status' in df.columns:
            st.subheader("📘 Distribusi Status Novel")
            status_counts = df['status'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
            ax2.axis('equal')
            st.pyplot(fig2)
        else:
            st.warning("Kolom 'status' tidak ditemukan dalam dataset.")
