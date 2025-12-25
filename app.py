import streamlit as st
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    page_icon="🌸",
    layout="centered"
)

# =========================
# CSS 
# =========================
st.markdown("""
<style>
body {
    background-color: #fff0f6;
}

.main {
    background-color: #fff0f6;
}

h1, h2, h3 {
    color: #ad1457;
}

.stButton > button {
    background-color: #f48fb1;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #ec407a;
}

.stTextArea textarea {
    border-radius: 12px;
}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.stProgress > div > div {
    background-color: #f06292;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LABEL
# =========================
LABEL_MAP = {
    0: "Negatif 😡",
    1: "Netral 😐",
    2: "Positif 😊"
}

# =========================
# LOAD MODELS 
# =========================
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("Model Bert Base Uncased/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("Model Bert Base Uncased/model")
    return tokenizer, model

@st.cache_resource
def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained("Model DistilBert/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("Model DistilBert/model")
    return tokenizer, model

@st.cache_resource
def load_lstm():
    model = load_model("Model LSTM/model_lstm.keras")
    with open("Model LSTM/tokenizer_lstm.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# =========================
# FUNGSI PREDIKSI
# =========================
def predict_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    return label, probs.detach().numpy()

def predict_lstm(text, tokenizer, model):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    probs = model.predict(padded)
    label = np.argmax(probs)
    return label, probs

# =========================
# HEADER
# =========================
st.markdown("<h1 style='text-align:center;'>🌸 Dashboard Klasifikasi Sentimen Opini Publik di Twitter Terhadap Kebijakan Pemerintah 🌸</h1>", unsafe_allow_html=True)


# =========================
# INPUT 
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

text = st.text_area(
    "📝 Masukkan Teks",
    placeholder="Contoh: Pelayanan aplikasi ini sangat memuaskan...",
    height=140
)

model_choice = st.selectbox(
    "🤖 Pilih Model",
    ["BERT Base Uncased", "DistilBERT", "LSTM"]
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TOMBOL PREDIKSI
# =========================
if st.button("🔍 Prediksi Sentimen"):
    if text.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        with st.spinner("⏳ Sedang memproses..."):
            if model_choice == "BERT Base Uncased":
                tokenizer, model = load_bert()
                label, prob = predict_bert(text, tokenizer, model)
            elif model_choice == "DistilBERT":
                tokenizer, model = load_distilbert()
                label, prob = predict_bert(text, tokenizer, model)
            else:
                model, tokenizer = load_lstm()
                label, prob = predict_lstm(text, tokenizer, model)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.success("✅ Prediksi berhasil")

        label_text = LABEL_MAP[label]

        st.subheader("🧾 Hasil Prediksi")
        st.write(f"**Label Angka:** `{label}`")
        st.write(f"**Sentimen:** **{label_text}**")

        st.subheader("📊 Probabilitas")
        probs = prob[0]
        for i, p in enumerate(probs):
            st.write(f"{LABEL_MAP[i]} : **{p:.2%}**")
            st.progress(float(p))

        st.markdown("</div>", unsafe_allow_html=True)
