import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import zipfile
import io

st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# ------------------ Upload dan Load Data ------------------
@st.cache_data
def load_data_from_zip(zip_file):
    with zipfile.ZipFile(zip_file) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            st.error("❌ File .csv tidak ditemukan dalam ZIP.")
            st.stop()
        with z.open(csv_files[0]) as f:
            return pd.read_csv(f)

st.sidebar.title("📁 Upload Dataset ZIP")
uploaded_file = st.sidebar.file_uploader("Unggah file ZIP yang berisi 'novels.csv'", type="zip")

if uploaded_file:
    df = load_data_from_zip(uploaded_file)
else:
    st.warning("📌 Silakan unggah file ZIP terlebih dahulu.")
    st.stop()

# ------------------ Inisialisasi Riwayat ------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------ CSS Styling ------------------
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

# ------------------ Sidebar Navigasi ------------------
page = st.sidebar.radio("📚 Navigasi", ["🏠 Home", "⭐ Rekomendasi Scored", "🎯 Rekomendasi Genre", "📊 Distribusi Novel"])

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
    st.markdown("Masukkan skor dan sistem akan merekomendasikan novel dengan **scored serupa** menggunakan algoritma **Random Forest Regressor**.")

    input_score = st.slider("🎯 Pilih Nilai Skor", 
                            min_value=float(df['scored'].min()),
                            max_value=float(df['scored'].max()), 
                            value=float(df['scored'].mean()), 
                            step=0.01)

    X = df[['scored']]
    y = df['popularty']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    r2_score = model.score(X, y)
    st.markdown(f"📈 <b>Model R² Score:</b> <code>{r2_score:.4f}</code>", unsafe_allow_html=True)

    predicted_pop = model.predict([[input_score]])[0]
    st.markdown(f"📊 <b>Prediksi Popularitas untuk skor {input_score:.2f}:</b> <code>{predicted_pop:.2f}</code>", unsafe_allow_html=True)

    df['predicted_popularty'] = model.predict(df[['scored']])
    df['predicted_diff'] = abs(df['predicted_popularty'] - predicted_pop)
    recommended = df.sort_values(by='predicted_diff').head(5)

    st.markdown("### 📚 Rekomendasi Novel:")
    st.dataframe(recommended[['title', 'authors', 'genres', 'scored', 'popularty', 'predicted_popularty']], use_container_width=True)

    st.session_state.history.append({
        'judul_dipilih': f'Scored {input_score:.2f}',
        'metode': 'random_forest_scored',
        'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
    })

# ------------------ Rekomendasi Berdasarkan Genre dari Judul ------------------
elif page == "🎯 Rekomendasi Genre":
    st.title("🎯 Rekomendasi Novel Berdasarkan Genre dari Judul")
    st.markdown("Masukkan judul novel, dan sistem akan menampilkan rekomendasi novel dengan genre yang sama.")

    title_input = st.text_input("✏️ Masukkan Judul Novel (case-sensitive)")

    if title_input:
        selected_novel = df[df['title'] == title_input]

        if not selected_novel.empty:
            selected_genre = selected_novel.iloc[0]['genres']
            genre_novels = df[df['genres'] == selected_genre]

            st.markdown(f"### 📌 Genre: <span style='color:green'><code>{selected_genre}</code></span>", unsafe_allow_html=True)

            X_genre = genre_novels[['scored']]
            y_genre = genre_novels['popularty']
            model_genre = RandomForestRegressor(n_estimators=100, random_state=42)
            model_genre.fit(X_genre, y_genre)

            r2_genre = model_genre.score(X_genre, y_genre)
            st.markdown(f"📈 <b>Model R² Score (genre ini):</b> <code>{r2_genre:.4f}</code>", unsafe_allow_html=True)

            genre_novels['predicted_popularty'] = model_genre.predict(X_genre)
            recommended = genre_novels.sort_values(by='predicted_popularty', ascending=False).head(5)

            st.markdown("### 📚 Rekomendasi Novel Berdasarkan Prediksi Popularitas:")
            st.dataframe(recommended[['title', 'authors', 'genres', 'scored', 'popularty', 'predicted_popularty']], use_container_width=True)

            st.session_state.history.append({
                'judul_dipilih': title_input,
                'metode': 'genre + random_forest',
                'rekomendasi': recommended[['title', 'authors', 'genres', 'scored']]
            })
        else:
            st.warning("Judul tidak ditemukan dalam data.")

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
