import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
import json
import random

# ======================
# KONFIGURASI HALAMAN
# ======================
st.set_page_config(
    page_title="Regression",
    layout="wide",
    page_icon=":notebook:"
)

# ======================
# SIDEBAR (DROPDOWN)
# ======================
st.sidebar.title("📌 Navigasi")

menu = st.sidebar.selectbox(
    "Pilih Halaman",
    (
        "Materi Pembelajaran",
        "Infografis",
        "Audio",
        "Video",
        "Dataset Tomat",
        "Dataset Sekolah",
        "Dataset Borobudur",
        "AI-powered Quiz",
        "Code Review"
    )
)

quotes = [
    ("Man jadda wajada — siapa yang bersungguh-sungguh pasti berhasil.", "Pepatah Arab"),
    ("Belajar hari ini adalah investasi untuk masa depan.", "Anonim"),
    ("Kesuksesan adalah hasil dari kebiasaan kecil yang dilakukan setiap hari.", "James Clear"),
    ("Jangan takut salah, takutlah jika tidak pernah mencoba.", "Buya Hamka"),
    ("Ilmu tidak akan datang kepada orang yang malas.", "Imam Syafi’i"),
    ("Proses tidak pernah mengkhianati hasil.", "Anonim"),
    ("Sedikit demi sedikit, lama-lama menjadi mahir.", "Pepatah Indonesia"),
    ("Belajar bukan untuk hari ini saja, tapi untuk seumur hidup.", "Ki Hajar Dewantara"),
    ("Gagal itu biasa, berhenti itu pilihan.", "Anonim"),
    ("Kerja keras mengalahkan bakat jika bakat tidak bekerja keras.", "Tim Notke")
]


quote, author = random.choice(quotes)
st.sidebar.success(f'"{quote}" -- {author}')
st.sidebar.caption("Dibuat dengan :fire: oleh **Adi Setiawan**")
# ======================
# HALAMAN UTAMA
# ======================
# st.title("🎓 Aplikasi Pembelajaran")

# ======================
# ISI HALAMAN
# ======================

# 1. MATERI PEMBELAJARAN
if menu == "Materi Pembelajaran":
    st.header("📘 Materi Pembelajaran")
    st.markdown("""
### 1. Pengantar Machine Learning Regression
Regression adalah metode Supervised Learning yang digunakan untuk memprediksi nilai kontinu berdasarkan satu atau lebih variabel input (fitur).
Tujuan regression adalah **menemukan hubungan matematis antara variabel input (X) dan target (y)**.

Contoh kasus penggunaan:
- Prediksi harga rumah
- Prediksi penjualan
- Prediksi jumlah produksi
- Prediksi pendapatan

### 2. Jenis-Jenis Algoritma Regression
Algoritma regression yang umum digunakan:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree Regression
6. Random Forest Regression
7. Support Vector Regression (SVR)

Pada materi ini digunakan Linear Regression dan Ridge Regression.

### 3. Library yang Digunakan
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
```

### 4. Load Dataset dengan Pandas
```python
df = pd.read_csv("data_penjualan.csv")
df.head()
```

Contoh struktur dataset:
```python
Iklan | Penjualan
10    | 25
20    | 45
30    | 65
```

### 5. Exploratory Data Analysis (EDA)
Visualisasi scatter plot digunakan untuk melihat hubungan antara input dan target.
```python
plt.scatter(df["Iklan"], df["Penjualan"])
plt.xlabel("Biaya Iklan")
plt.ylabel("Penjualan")
plt.title("Hubungan Iklan vs Penjualan")
plt.show()
```

### 6. Split Data Train dan Test
```python
X = df[["Iklan"]]
y = df["Penjualan"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 7. Membuat Model Linear Regression
```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

Evaluasi model:
```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```
### 8. Hyperparameter Tuning dengan GridSearchCV
```python
ridge = Ridge()
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(ridge, param_grid, cv=5, scoring="r2")
grid.fit(X_train, y_train)
```

### 9. Model Terbaik
```python
best_model = grid.best_estimator_
best_params = grid.best_params_
```
### 10. Evaluasi Model Terbaik
```python
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
```
### 11. Menyimpan Model Terbaik
```python
joblib.dump(best_model, "model_regression.pkl")
model = joblib.load("model_regression.pkl")
```

### 12. Deployment Model ke Streamlit
```python
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model_regression.pkl")
st.title("Aplikasi Prediksi Penjualan")
iklan = st.number_input("Masukkan biaya iklan", min_value=0)

if st.button("Prediksi"):
    prediksi = model.predict([[iklan]])
    st.success(f"Prediksi penjualan: {prediksi[0]:.2f}")
``` 

### 13. Alur Lengkap Machine Learning Regression
1. Load dataset
2. EDA dan visualisasi
3. Split data
4. Training model
5. Evaluasi
6. Hyperparameter tuning
7. Simpan model terbaik
8. Deploy ke Streamlit

### 14. Kesimpulan
Regression digunakan untuk prediksi nilai kontinu.
Visualisasi penting sebelum modeling.
GridSearchCV membantu menemukan parameter terbaik.
Streamlit memudahkan deployment model ML.
    """)

# 2. INFOGRAFIS
elif menu == "Infografis":
    st.header("📊 Infografis")

    st.image("Infografis.png", caption="Infografis Machine Learning Regression")
    
    st.image("Infografis 2.png", caption="Infografis Machine Learning Regression")
    
    st.image("Infografis 3.png", caption="Berbagai Algoritma Machine Learning Regression")
    
    st.image("Infografis 4.png", caption="Alur Kerja Machine Learning Regression")

# 3. AUDIO
elif menu == "Audio":
    st.header("🎧 Audio Pembelajaran")
    st.write("Podcast tentang Machine Learning Regression")

    st.audio(
        "Bedah Konsep Machine Learning Regression.m4a"
    )

# 4. VIDEO
elif menu == "Video":
    st.header("🎬 Video Pembelajaran")
    st.write("Contoh skenario penerapan Machine Learning Regression dalam Penjualan Bakso")

    st.video(
        "Video Machine Learning Regression.mp4"
    )

# 5. DATASET FOLLOWER
elif menu == "Dataset Borobudur":
    st.header("🛕 Dataset Borobudur")
    st.image("Ilustrasi Borobudur.png", caption="Ilustrasi")
    st.markdown("""
    Dataset Borobudur ini menggambarkan jumlah kunjungan wisatawan ke Candi Borobudur yang dicatat setiap hari oleh pengelola kawasan wisata. Jumlah pengunjung dipengaruhi oleh beberapa faktor sederhana seperti **hari kunjungan** (hari biasa atau akhir pekan), **kondisi cuaca**, serta adanya **promo tiket** atau acara khusus. 
    
    Melalui dataset ini, siswa dapat memahami bahwa data tidak hanya berasal dari laboratorium atau perusahaan teknologi, tetapi juga dari aktivitas pariwisata yang dekat dengan kehidupan sehari-hari. Dengan menganalisis pola kunjungan menggunakan machine learning, siswa bisa mempelajari cara memprediksi jumlah wisatawan agar pengelola dapat mengatur jumlah petugas, tiket, dan fasilitas secara lebih efisien.
    """)
    # Sample DataFrame (per hari)
    df = pd.DataFrame({
        "Tanggal": pd.date_range(start="2023-01-01", periods=14, freq="D"),
        "Hari": [
            "Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu",
            "Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"
        ],
        "Cuaca": [
            "Cerah", "Cerah", "Hujan", "Cerah", "Berawan", "Cerah", "Cerah",
            "Berawan", "Hujan", "Cerah", "Cerah", "Berawan", "Hujan", "Cerah"
        ],
        "Promo": [
            "Tidak", "Tidak", "Tidak", "Ya", "Tidak", "Tidak", "Ya",
            "Ya", "Tidak", "Tidak", "Ya", "Tidak", "Tidak", "Ya"
        ],
        "Jumlah Pengunjung": [
            14500, 8200, 6100, 9000, 8800, 9400, 15200,
            13800, 7900, 8300, 9100, 8600, 6400, 15800
        ]
    })

    st.subheader("Data Kunjungan Harian Candi Borobudur")
    st.dataframe(df)

    st.subheader("Tren Jumlah Pengunjung Harian")
    st.line_chart(df.set_index("Tanggal")["Jumlah Pengunjung"])

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Dataset Borobudur Harian (CSV)",
        data=csv,
        file_name="dataset_borobudur_harian.csv",
        mime="text/csv", type="primary"
    )

# 6. DATASET PENJUALAN TOMAT
elif menu == "Dataset Tomat":
    st.header("🍅 Dataset Penjualan Tomat (Harian)")
    
    st.image("Ilustrasi Tomat.png", caption="Ilustrasi")

    # Sample DataFrame dengan fitur sederhana
    # Dataset dengan fitur sederhana & realistis
    df = pd.DataFrame({
        "Harga Tomat (Rp/kg)": [
            8000, 9000, 10000, 11000, 12000,
            13000, 14000, 15000, 16000, 17000
        ],
        "Cuaca": [
            "Cerah", "Cerah", "Berawan", "Hujan", "Cerah",
            "Berawan", "Hujan", "Cerah", "Berawan", "Hujan"
        ],
        "Promo": [
            "Ya", "Tidak", "Tidak", "Tidak", "Ya",
            "Tidak", "Tidak", "Ya", "Tidak", "Tidak"
        ],
        "Penjualan (Kg)": [
            220, 210, 195, 170, 205,
            190, 160, 200, 185, 155
        ]
    })

    st.subheader("Data Penjualan Tomat Harian")
    st.markdown("""
    **Dataset penjualan tomat** ini diambil dari aktivitas harian seorang pedagang sayur di pasar tradisional yang memasok tomat segar ke warung makan dan rumah tangga di sekitarnya. Setiap hari, jumlah tomat yang terjual dipengaruhi oleh beberapa faktor sederhana seperti **cuaca (apakah hujan atau cerah), hari (hari sekolah atau akhir pekan), serta promo harga yang diberikan untuk menarik pembeli**. 
    
    Dengan dataset ini, siswa dapat melihat bagaimana data penjualan nyata berubah dari hari ke hari dan memahami bahwa konsep machine learning tidak hanya digunakan di perusahaan besar, tetapi juga bisa membantu pedagang kecil memprediksi stok tomat agar tidak rugi karena busuk atau kehabisan barang.
    """)
    st.dataframe(df)
    st.subheader("Hubungan Harga Tomat dan Penjualan")
    st.line_chart(
        df.set_index("Harga Tomat (Rp/kg)")["Penjualan (Kg)"]
    )

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Dataset Penjualan Tomat (CSV)",
        data=csv,
        file_name="dataset_penjualan_tomat_harian.csv",
        mime="text/csv", type="primary"
    )

# 7. DATASET PENJUALAN KOPI
elif menu == "Dataset Sekolah":
    st.header("🎓 Dataset Sekolah – Prediksi Nilai TKA Matematika")
    st.image("Ilustrasi Sekolah.png", caption="Ilustrasi")
    st.markdown("""
    Dataset sekolah ini menggambarkan kondisi belajar siswa kelas XI di sebuah SMK, di mana nilai TKA Matematika dipengaruhi oleh kebiasaan belajar sehari-hari. Setiap siswa memiliki **jam belajar** yang berbeda di rumah, **tingkat kehadiran** di sekolah, serta sebagian mengikuti **bimbingan belajar (bimbel)** untuk memperkuat pemahaman materi. 
    
    Dengan dataset ini, siswa dapat melihat bahwa nilai ujian bukan hanya soal “pintar atau tidak”, tetapi merupakan hasil dari kombinasi usaha, kedisiplinan, dan strategi belajar. Melalui analisis dan pemodelan machine learning, siswa dapat memprediksi nilai TKA Matematika dan memahami faktor mana yang paling berpengaruh terhadap hasil belajar mereka.
    """)

    # Sample DataFrame (fitur sederhana)
    df = pd.DataFrame({
        "Jam Belajar (jam/hari)": [
            1, 2, 3, 4, 2.5, 3.5, 4.5,
            1.5, 2, 3, 4, 5, 3.5, 4.5
        ],
        "Kehadiran (%)": [
            80, 85, 90, 95, 88, 92, 98,
            82, 87, 91, 94, 100, 93, 99
        ],
        "Bimbel": [
            "Tidak", "Tidak", "Ya", "Ya", "Tidak", "Ya", "Ya",
            "Tidak", "Tidak", "Ya", "Ya", "Ya", "Ya", "Ya"
        ],
        "Nilai TKA Matematika": [
            60, 68, 75, 82, 70, 85, 90,
            65, 69, 78, 84, 92, 88, 95
        ]
    })

    st.subheader("Data Siswa")
    st.dataframe(df)

    st.subheader("Hubungan Jam Belajar dan Nilai TKA Matematika")
    st.line_chart(
        df.set_index("Jam Belajar (jam/hari)")["Nilai TKA Matematika"]
    )

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Dataset Sekolah (CSV)",
        data=csv,
        file_name="dataset_sekolah_tka_matematika.csv",
        mime="text/csv", type="primary"
    )


# 8. AI-POWERED QUIZ
elif menu == "AI-powered Quiz":
    # ===== Ambil HF token dari Streamlit secrets =====
    HF_TOKEN = st.secrets["HF_TOKEN"]
    client = InferenceClient(token=HF_TOKEN)

    # ===== Model Qwen terbaru =====
    MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # ===== Load Excel quiz =====
    @st.cache_data
    def load_quiz():
        df = pd.read_excel("post-test.xlsx")  # File harus ada di folder sama
        df = df.reset_index(drop=True)
        df.index += 1  # Nomor soal mulai dari 1
        return df

    df = load_quiz()

    st.subheader(":brain: Machine Learning Regression Quiz")
    st.markdown(f"with Personalized Feedback Powered by `{MODEL}` | Total : {len(df)} Questions")

    # ===== Form jawaban siswa =====
    with st.form("quiz_form"):
        jawaban_siswa = {}
        for idx, row in df.iterrows():
            st.markdown(f"<b>Soal {idx} : {row['soal']}</b>", unsafe_allow_html=True)
            options = [row['opsi_a'], row['opsi_b'], row['opsi_c'], row['opsi_d'], row['opsi_e']]
            labels = ["A", "B", "C", "D", "E"]
            jawaban = st.radio("Pilih jawaban:", options, key=f"soal_{idx}", index=None)
            if jawaban:
                jawaban_siswa[idx] = labels[options.index(jawaban)]

        submitted = st.form_submit_button("Submit Jawaban", type="primary")

    # ===== Analisis jawaban =====
    if submitted:
        if len(jawaban_siswa) < len(df):
            st.warning("Belum semua soal dijawab! Silakan lengkapi.")
        else:
            skor = 0
            salah = []
            bab_lemah = {}

            for idx, row in df.iterrows():
                if jawaban_siswa[idx] == row['jawaban_benar']:
                    skor += 1
                else:
                    salah.append({
                        "no": idx,
                        "soal": row['soal'],
                        "jawaban_siswa": jawaban_siswa[idx],
                        "jawaban_benar": row['jawaban_benar'],
                        "bab": row['bab']
                    })
                    bab_lemah[row['bab']] = bab_lemah.get(row['bab'], 0) + 1

            total = len(df)
            persentase = (skor / total) * 100
            
            if persentase < 80:
                st.snow()
            else:
                st.balloons()
            
            st.success(f"Skor kamu: {skor}/{total} ({persentase:.1f}%)")

            if salah:
                st.error("Soal yang salah:")
                for s in salah:
                    st.write(f"**No {s['no']}** ({s['bab']}): {s['soal'][:100]}... | Jawabanmu: {s['jawaban_siswa']} → Benar: {s['jawaban_benar']}")

                # ===== Generate personalized feedback via Qwen =====
                with st.spinner("Sedang generate feedback personal dari AI..."):
                    ringkasan = f"""
    Siswa skor {skor}/{total} ({persentase:.1f}%).
    Jumlah salah: {len(salah)}.
    Bab lemah (jumlah salah): {json.dumps(dict(sorted(bab_lemah.items(), key=lambda x: x[1], reverse=True)), ensure_ascii=False)}.
    Contoh soal salah: {json.dumps(salah[:5], ensure_ascii=False, indent=2)}
    """

                    response = client.chat_completion(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": "Kamu adalah guru Machine Learning yang ramah dan motivatif TAPI SINGKAT DAN PADAT untuk siswa SMK kelas 11 di Indonesia. Berikan feedback gaya sandwich: mulai positif (puji yang benar), lalu konstruktif (sebut bab lemah spesifik dan rekomendasi belajar), akhiri positif (dorong semangat). Gunakan bahasa Indonesia yang edukatif. Panggil dengan sapaan 'Halo Siswa Hebat!'. Akhiri dengan kalimat 'Jangan Pernah Berputus Asa :fire:!"},
                            {"role": "user", "content": f"Beri personalized feedback berdasarkan hasil quiz regression ini:\n{ringkasan}\nRekomendasikan materi atau latihan spesifik untuk bab lemah."}
                        ],
                        max_tokens=700,
                        temperature=0.5
                    )

                    feedback = response.choices[0].message.content
                    st.markdown("### Personalized Feedback dari AI Guru:")
                    st.info(feedback)
            else:
                st.balloons()
                st.success(":fire: Perfect! Kamu menguasai semua materi. Lanjut ke praktik coding ya!")


# 9. CODE REVIEW
elif menu == "Code Review":
    st.header("🧠 Code Review – Machine Learning")

    st.markdown("""
    ### 📝 Instruksi Soal
    **Tuliskan kode program Python untuk melakukan `train_test_split`**
    dengan ketentuan berikut:
    - Gunakan library `scikit-learn`
    - import modul train_test_split
    - `test_size = 0.2`
    - Gunakan `random_state`
    - Asumsikan data sudah ada di variabel `X` dan `y`
    """)
    
    # ===== Text Area Input Kode =====
    kode_user = st.text_area(
        "✍️ Tulis kode Python kamu di sini:",
        height=300,
        placeholder="""
from sklearn.model_selection import train_test_split

# tulis kode kamu di sini
"""
    )
    
    # ===== Tombol Review =====
    review_clicked = st.button("Code Review", type="primary")

    # ===== Setup HuggingFace Client =====
    HF_TOKEN = st.secrets["HF_TOKEN"]
    client = InferenceClient(token=HF_TOKEN)

    MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # ===== Proses Review =====
    if review_clicked:
        if kode_user.strip() == "":
            st.warning("⚠️ Kode belum diisi.")
        else:
            with st.spinner("🤖 AI sedang melakukan code review..."):
                prompt = f"""Kamu adalah guru Machine Learning untuk siswa SMK kelas 11 di Indonesia.
                Tugas siswa:"Tuliskan kode program untuk melakukan train test split dengan test_size = 0.2."
                Kode siswa:```python {kode_user} 
                Lakukan:
                Cek apakah kodenya sudah BENAR
                Jelaskan jika ada kesalahan
                Berikan versi kode yang BENAR
                Jelaskan fungsi setiap baris secara singkat
                Gunakan bahasa Indonesia yang jelas dan edukatif.
                """

            response = client.chat_completion(
                     model=MODEL,
                     messages=[
                         {"role": "system", "content": "Kamu adalah guru ML yang sabar, jelas, dan mendidik."},
                         {"role": "user", "content": prompt}
                     ],
                     max_tokens=600,
                     temperature=0.4
                 )

            hasil_review = response.choices[0].message.content

            st.subheader("📋 Hasil Code Review dari AI")
            st.info(hasil_review)
         
