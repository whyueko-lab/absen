import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(page_title="Prediksi Keterlambatan Karyawan", layout="wide")
st.title("🕒 Aplikasi Prediksi Keterlambatan Karyawan")

# -----------------------------
# Sidebar Navigasi
# -----------------------------
with st.sidebar:
    st.title("📁 Menu Navigasi")
    menu = st.radio("Pilih Halaman:", ["📊 Dataset", "🧠 Prediksi", "📌 Evaluasi Model"])
    st.markdown("---")
    st.markdown("✨ Dibuat Oleh Rizky Septa Renaldy")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset_absensi_dengan_jam.xlsx")

df = load_data()

# -----------------------------
# Membuat Label Keterlambatan
# -----------------------------
df['Status Keterlambatan'] = df['Waktu Masuk (jam)'].apply(lambda x: 'Terlambat' if x > 8 else 'Tidak Terlambat')

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(df):
    df_encoded = df.copy()
    encoders = {}
    for col in ['Jenis Kelamin', 'Status Perkawinan', 'Transportasi', 'Cuaca', 'Status Keterlambatan']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    fitur_model = ['Usia', 'Jenis Kelamin', 'Status Perkawinan', 'Transportasi', 'Cuaca',
                   'Waktu Masuk (jam)', 'Rata-rata Jam Tidur']
    X = df_encoded[fitur_model]
    y = df_encoded["Status Keterlambatan"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, encoders, scaler, fitur_model

X, y, encoders, scaler, fitur_model = preprocess(df)

# -----------------------------
# Training Model + SMOTE
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Validasi jumlah kelas
if len(np.unique(y_train)) < 2:
    st.error("❌ Data latih hanya memiliki satu kelas. SMOTE membutuhkan minimal dua kelas berbeda.")
    st.stop()

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# Sidebar Info Distribusi Label
# -----------------------------
original_dist = collections.Counter(y_train)
resampled_dist = collections.Counter(y_train_res)

st.sidebar.markdown("### ⚖️ Distribusi Label")
st.sidebar.write("Sebelum SMOTE:", dict(original_dist))
st.sidebar.write("Sesudah SMOTE:", dict(resampled_dist))

# -----------------------------
# Halaman Dataset
# -----------------------------
if menu == "📊 Dataset":
    st.subheader("📊 Tabel Dataset Karyawan")
    st.dataframe(df, use_container_width=True)

    st.subheader("🔍 Distribusi Keterlambatan")
    fig, ax = plt.subplots()
    sns.countplot(x="Status Keterlambatan", data=df, palette="Set2", ax=ax)
    ax.set_title("Distribusi Status Keterlambatan")
    st.pyplot(fig)

    st.subheader("📈 Korelasi Antar Fitur")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title("Matriks Korelasi Fitur Numerik")
    st.pyplot(fig_corr)

# -----------------------------
# Halaman Prediksi
# -----------------------------
elif menu == "🧠 Prediksi":
    st.subheader("📝 Formulir Prediksi Keterlambatan")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", encoders['Jenis Kelamin'].classes_)
            status = st.selectbox("Status Perkawinan", encoders['Status Perkawinan'].classes_)
            transport = st.selectbox("Transportasi", encoders['Transportasi'].classes_)
        with col2:
            cuaca = st.selectbox("Cuaca", encoders['Cuaca'].classes_)
            usia = st.slider("Usia", 18, 60, 30)
            jam_tidur = st.slider("Rata-rata Jam Tidur (jam)", 0.0, 12.0, 6.0)
            jam_masuk = st.slider("Waktu Masuk (jam)", 0, 23, 8)

        submit = st.form_submit_button("🔍 Prediksi Sekarang")

    if submit:
        input_df = pd.DataFrame([[ 
            usia,
            encoders['Jenis Kelamin'].transform([jenis_kelamin])[0],
            encoders['Status Perkawinan'].transform([status])[0],
            encoders['Transportasi'].transform([transport])[0],
            encoders['Cuaca'].transform([cuaca])[0],
            jam_masuk,
            jam_tidur
        ]], columns=fitur_model)

        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        hasil = encoders['Status Keterlambatan'].inverse_transform([pred])[0]

        if hasil.lower() == "terlambat":
            st.error("⏰ **Prediksi: TERLAMBAT**\n\n📌 Karyawan kemungkinan akan datang *terlambat*. Periksa faktor cuaca, transportasi, atau waktu tidur.")
        else:
            st.success("✅ **Prediksi: TIDAK TERLAMBAT**\n\n👍 Karyawan diprediksi akan *tepat waktu*. Pertahankan rutinitas baik ini.")

# -----------------------------
# Halaman Evaluasi
# -----------------------------
elif menu == "📌 Evaluasi Model":
    st.subheader("📊 Evaluasi Kinerja Model")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Akurasi Model", f"{acc*100:.2f}%")
    with col2:
        st.write(f"📦 Jumlah Data Latih (SMOTE): **{len(X_train_res)}**")

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=encoders['Status Keterlambatan'].classes_,
                yticklabels=encoders['Status Keterlambatan'].classes_,
                ax=ax_cm)
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    st.subheader("💡 Fitur Paling Berpengaruh")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Fitur": fitur_model, "Pentingnya": importances})
    feat_df = feat_df.sort_values(by="Pentingnya", ascending=False)
    fig_feat, ax_feat = plt.subplots()
    sns.barplot(x="Pentingnya", y="Fitur", data=feat_df, palette="viridis", ax=ax_feat)
    ax_feat.set_title("Feature Importance")
    st.pyplot(fig_feat)

    with st.expander("📋 Laporan Klasifikasi Lengkap"):
        report = classification_report(y_test, y_pred, target_names=encoders['Status Keterlambatan'].classes_)
        st.code(report, language='text')
