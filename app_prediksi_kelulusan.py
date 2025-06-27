import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset langsung di dalam kode
data = {
    "IPK": [3.28,3.35,2.44,2.84,2.56,2.19,3.95,3.41,2.76,2.42,
            3.5,2.3,2.8,3.9,2.1,3.2,2.6,3.7,2.5,3.1,
            2.2,3.8,2.9,2.4,3.6,2.7,3.85,2.35,3.0,2.45],
    "Jumlah_SKS": [83,149,144,137,107,124,128,85,138,114,
                   144,110,120,150,100,130,115,140,125,135,
                   105,145,118,112,143,122,148,108,132,110],
    "IPS_Terakhir": [3.48,2.17,3.2,3.18,3.91,3.21,2.16,3.32,3.27,3.4,
                     3.6,2.1,2.5,3.8,2.0,3.0,2.9,3.7,2.4,3.2,
                     2.1,3.9,2.8,2.3,3.75,2.6,3.85,2.2,3.1,2.0],
    "Mengulang": [1,3,4,0,2,0,2,2,2,5,
                  0,3,1,0,4,1,3,0,5,2,
                  4,0,2,4,0,2,0,3,1,5],
    "Cuti": ["Iya","Iya","Iya","Iya","Iya","Tidak","Tidak","Iya","Iya","Iya",
             "Tidak","Iya","Iya","Tidak","Iya","Tidak","Iya","Tidak","Iya","Tidak",
             "Iya","Tidak","Iya","Iya","Tidak","Iya","Tidak","Iya","Tidak","Iya"],
    "Semester": [6,4,14,10,5,12,13,7,9,13,
                 8,14,9,8,15,10,11,8,14,9,
                 13,8,10,13,8,11,8,14,9,15],
    "Lama_Studi": [4,4,7,6,7,4,5,4,6,5,
                   4,7,6,4,7,5,6,4,7,5,
                   7,4,6,7,4,6,4,7,5,7],
    "Status": ["Lulus","Lulus","Tidak Lulus","Lulus Tidak Tepat Waktu","Lulus Tidak Tepat Waktu",
               "Tidak Lulus","Lulus Tidak Tepat Waktu","Lulus","Lulus Tidak Tepat Waktu","Tidak Lulus",
               "Lulus","Tidak Lulus","Lulus Tidak Tepat Waktu","Lulus","Tidak Lulus","Lulus",
               "Lulus Tidak Tepat Waktu","Lulus","Tidak Lulus","Lulus",
               "Tidak Lulus","Lulus","Lulus Tidak Tepat Waktu","Tidak Lulus","Lulus",
               "Lulus Tidak Tepat Waktu","Lulus","Tidak Lulus","Lulus","Tidak Lulus"]
}

# Load Data
df = pd.DataFrame(data)
df["Cuti"] = df["Cuti"].map({"Tidak": 0, "Iya": 1})
df["Status_Label"] = df["Status"].map({
    "Tidak Lulus": 0,
    "Lulus Tidak Tepat Waktu": 1,
    "Lulus": 2
})

X = df.drop(["Status", "Status_Label"], axis=1)
y = df["Status_Label"]

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Streamlit Interface
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")

st.markdown("Prediksi kelulusan mahasiswa berdasarkan data akademik.")
st.markdown("""
### 🧠 Tentang Model Prediksi

Model yang digunakan untuk memprediksi status kelulusan mahasiswa adalah **Decision Tree Classifier**.

Model ini bekerja dengan **membagi data berdasarkan aturan-aturan sederhana** seperti IPK, jumlah SKS, jumlah mata kuliah yang diulang, riwayat cuti, serta indikator lainnya untuk menentukan apakah mahasiswa akan:
- ✅ **Lulus tepat waktu**
- ⏱️ **Lulus tidak tepat waktu**
- ❌ **Tidak lulus**

Model ini dilatih menggunakan data akademik historis mahasiswa, dan mampu membuat keputusan berdasarkan **kombinasi berbagai faktor** yang memengaruhi keberhasilan studi mahasiswa.

Visualisasi pohon keputusan di bawah ini memperlihatkan bagaimana setiap keputusan dibuat oleh model berdasarkan data input. Hasil prediksi juga dilengkapi dengan **probabilitas** untuk membantu memahami seberapa besar tingkat keyakinan model terhadap prediksinya.

---

### 📌 Faktor yang Paling Mempengaruhi Kelulusan Mahasiswa

Berdasarkan hasil pemodelan, beberapa faktor utama yang paling berpengaruh terhadap kelulusan mahasiswa di antaranya:

- **IPK (Indeks Prestasi Kumulatif):** Mewakili performa akademik keseluruhan.
- **IPS Terakhir:** Indikator tren nilai terbaru, menunjukkan apakah ada peningkatan atau penurunan.
- **Jumlah SKS:** Jumlah beban studi yang telah ditempuh.
- **Jumlah Mata Kuliah Mengulang:** Menunjukkan adanya hambatan akademik.
- **Riwayat Cuti:** Cuti akan memperpanjang masa studi atau menjadi indikasi kendala akademik/pribadi.
- **Semester Saat Ini:** Jika semester melebihi waktu studi wajar (misalnya 8 semester), kemungkinan kelulusan tepat waktu menurun.


Semakin besar pengaruh faktor tersebut pada hasil prediksi model, maka semakin tinggi perannya dalam menentukan kelulusan mahasiswa.
""")


with st.expander("📄 Lihat Dataset"):
    st.dataframe(df)

st.subheader("📝 Formulir Input")
with st.form("form_input"):
    ipk = st.slider("📌 IPK", 0.0, 4.0, 3.0, 0.01)
    sks = st.slider("📚 Jumlah SKS", 0, 160, 130)
    ips = st.slider("📈 IPS Terakhir", 0.0, 4.0, 3.0, 0.01)
    mengulang = st.slider("🔁 Mata Kuliah Mengulang", 0, 10, 1)
    cuti = st.selectbox("🛑 Pernah Cuti?", ["Tidak", "Iya"])
    semester = st.slider("📆 Semester Saat Ini", 1, 16, 8)
    lama_studi = st.slider("⏳ Lama Studi (tahun)", 1, 7, 4)
    tombol = st.form_submit_button("🔮 Prediksi")

if tombol:
    input_df = pd.DataFrame([{
        "IPK": ipk,
        "Jumlah_SKS": sks,
        "IPS_Terakhir": ips,
        "Mengulang": mengulang,
        "Cuti": 1 if cuti == "Iya" else 0,
        "Semester": semester,
        "Lama_Studi": lama_studi
    }])

    # Aturan manual tambahan
    if lama_studi == 7:
        hasil = "❌ Tidak Lulus (Masa studi maksimal 7 tahun tercapai)"
        proba = [1.0, 0.0, 0.0]
    elif semester > (lama_studi * 2):
        hasil = "⏱️ Lulus Tidak Tepat Waktu (Semester terlalu tinggi dibanding masa studi)"
        proba = [0.0, 1.0, 0.0]
    else:
        prediksi = clf.predict(input_df)[0]
        proba = clf.predict_proba(input_df)[0]
        label = {0: "❌ Tidak Lulus", 1: "⏱️ Lulus Tidak Tepat Waktu", 2: "✅ Lulus"}
        hasil = label[prediksi]

    # Hasil akhir
    st.subheader("📢 Hasil Prediksi")
    st.success(f"Mahasiswa diprediksi: **{hasil}**")

       # Visualisasi Persentase
    st.subheader("📊 Persentase Prediksi")
    persen_df = pd.DataFrame({
        "Status": ["❌ Tidak Lulus", "⏱️ Lulus Tidak Tepat Waktu", "✅ Lulus"],
        "Probabilitas": [round(p * 100, 2) for p in proba]
    })

    fig, ax = plt.subplots()
    sns.barplot(x="Status", y="Probabilitas", data=persen_df, palette="Set2", ax=ax)
    ax.set_ylabel("Persentase (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # Visualisasi Decision Tree
    st.subheader("🌳 Visualisasi Decision Tree")
    fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=X.columns, class_names=["Tidak Lulus", "Lulus Tidak Tepat Waktu", "Lulus"], filled=True)
    st.pyplot(fig_tree)
