import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import json

# ===== Streamlit page config =====
st.set_page_config(
    page_title="Personalized Feedback",
    page_icon=":robot:"
)

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

st.subheader(":chart_with_upwards_trend: Machine Learning Regression Quiz")
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
        st.success(f"Skor kamu: {skor}/{total} ({persentase:.1f}%)")

        if salah:
            st.warning("Soal yang salah:")
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
                        {"role": "system", "content": "Kamu adalah guru Machine Learning yang ramah dan motivatif untuk siswa SMK kelas 11 di Indonesia. Berikan feedback gaya sandwich: mulai positif (puji yang benar), lalu konstruktif (sebut bab lemah spesifik dan rekomendasi belajar), akhiri positif (dorong semangat). Gunakan bahasa Indonesia yang edukatif."},
                        {"role": "user", "content": f"Beri personalized feedback berdasarkan hasil quiz regression ini:\n{ringkasan}\nRekomendasikan materi atau latihan spesifik untuk bab lemah."}
                    ],
                    max_tokens=700,
                    temperature=0.7
                )

                feedback = response.choices[0].message.content
                st.markdown("### Personalized Feedback dari AI Guru:")
                st.info(feedback)
        else:
            st.balloons()
            st.success(":fire: Perfect! Kamu menguasai semua materi. Lanjut ke praktik coding ya!")

