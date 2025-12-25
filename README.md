
## 📌 Ringkasan Proyek

Proyek ini merupakan implementasi **Sistem Analisis Sentimen Teks** yang dibuat dalam bentuk website interaktif menggunakan Streamlit. Sistem ini digunakan untuk mengelompokkan teks, khususnya teks dari media sosial Twitter, ke dalam tiga kategori sentimen, yaitu negative, neutral, dan positive.

Pada proyek ini digunakan **tiga pendekatan model yang berbeda**, yaitu sebagai berikut :
* LSTM (Neural Network Base non-pretrained)
* BERT Base Uncased (Pretrained 1)
* DistilBERT (Pretrained 2)

Ketiga model tersebut dibandingkan untuk melihat perbedaan performa berdasarkan Accuracy, Precision, Recall, dan F1-Score, serta efisiensi model.

---

## 📂 Struktur Repository

Struktur folder pada repository disusun untuk memisahkan setiap model dan komponen utama sistem agar mudah dipahami dan dikelola.

```
Dashboard-Analisis Sentimen Teks/
│
├── .streamlit/
│   └── config.toml                        # Konfigurasi tampilan Streamlit
│
├── Dataset/       
│   ├── Twitter Sentiment Analysis.csv/    # Data CSV Twitter Sentiment Analysis 
│
├── Model Bert Base Uncased/               # Model BERT Base Uncased hasil fine-tuning
│   ├── model/                             # File model BERT 
│   └── tokenizer/                         # Tokenizer BERT
│
├── Model DistilBert/                      # Model DistilBERT hasil fine-tuning
│   ├── model/                             # File model DistilBERT
│   └── tokenizer/                         # Tokenizer DistilBERT
│
├── Model LSTM/                            # Model LSTM berbasis Neural Network
│   ├── model_lstm.keras                   # Model LSTM terlatih
│   └── tokenizer_lstm.pkl                 # Tokenizer Keras untuk LSTM
│
├── Src/                                   
│   ├── Twitter Sentiment Analysis.ipynb   # Program Pelatihan Model dan Evaluasi Model
│
├── app.py                                 # Program utama aplikasi Streamlit
└── README.md                              # Dokumentasi proyek
```

---

## 📊 Dataset yang Digunakan

Dataset yang digunakan pada proyek ini dari **Hugging Face Dataset Repository**, dengan nama dataset:

**Twitter Sentiment Analysis Dataset**
[https://huggingface.co/datasets/KidzRizal/twitter-sentiment-analysis](https://huggingface.co/datasets/KidzRizal/twitter-sentiment-analysis)

Dataset ini berisi kumpulan teks tweet yang telah diberi label sentimen sehingga sesuai untuk tugas **klasifikasi Sentimen Teks**.

### 📍 Informasi Dataset

```
RangeIndex: 8030 entries, 0 to 8029
Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype 
---  ------             --------------  ----- 
 0   original_text      8028 non-null   object
 1   final_text         7959 non-null   object
 2   sentiment          8030 non-null   int64 
 3   sentiment_label    8030 non-null   object
 4   __index_level_0__  8030 non-null   int64 
dtypes: int64(2), object(3)
```

### 📌 Penjelasan Kolom Dataset

| Kolom               | Deskripsi                                               |
| ------------------- | --------------------------------------------------------|
| `original_text`     | Teks tweet asli sebelum preprocessing                   |
| `final_text`        | Teks hasil preprocessing                                |
| `sentiment`         | Label numerik (0 = Negative, 1 = Neutral, 2 = Positive) |
| `sentiment_label`   | Label sentimen dalam bentuk teks                        |
| `__index_level_0__` | Indeks tambahan dari dataset                            |

### 📊 Jumlah Data

* **Total data:** 8.030 

---

## 🧹 Preprocessing Data

Tahapan preprocessing dengan langkah-langkah sebagai berikut:

### 1️⃣ Encoding Label

Label sentimen dalam bentuk teks diubah menjadi label numerik menggunakan **LabelEncoder**.

```python
le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment_label"])
```

Jumlah kelas yang dihasilkan:

```python
num_classes = df["label"].nunique()
print("Jumlah kelas:", num_classes)
```

Hasil: **3 kelas sentimen**

---

### 2️⃣ Penanganan Missing Value

Nilai kosong pada kolom `final_text` diisi dengan string kosong dan dikonversi menjadi tipe string.

```python
df["final_text"] = df["final_text"].fillna("").astype(str)
```

Langkah ini bertujuan untuk mencegah error saat proses tokenisasi.

---

### 3️⃣ Pembagian Data Train dan Test

Dataset dibagi menjadi **80% data latih** dan **20% data uji** , sehingga distribusi kelas tetap seimbang.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["final_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
```

---

## 🤖 Model yang Digunakan

### 1️⃣ LSTM (Long Short-Term Memory)

LSTM merupakan model *Neural Network Base* yang mempelajari urutan kata dalam teks. Model ini tidak menggunakan pretrained model dan dilatih sepenuhnya dari dataset.

**Karakteristik:**

* Non-pretrained
* Menggunakan Embedding Layer dan LSTM Layer
* Ringan dan cepat

**Kelebihan:**

* Komputasi rendah
* Cocok untuk baseline

**Kekurangan:**

* Pemahaman konteks terbatas
* Sangat bergantung pada preprocessing

---

### 2️⃣ BERT Base Uncased

BERT merupakan model **Transformer pretrained** yang memahami konteks kata secara dua arah. Model ini dilakukan **fine-tuning** untuk klasifikasi sentimen.

**Karakteristik:**

* Pretrained `bert-base-uncased`
* WordPiece Tokenizer
* Contextual embedding

**Kelebihan:**

* Akurasi tinggi
* Pemahaman konteks sangat baik

**Kekurangan:**

* Ukuran besar
* Resource komputasi tinggi

---

### 3️⃣ DistilBERT

DistilBERT adalah versi ringan dari BERT.

**Karakteristik:**

* Pretrained Transformer
* Lebih ringan dari BERT
* Dilakukan fine-tuning

**Kelebihan:**

* Lebih cepat dan efisien
* Akurasi mendekati BERT

**Kekurangan:**

* Sedikit lebih rendah dari BERT Base

---

## 📈 Hasil Evaluasi dan Analisis Perbandingan

Evaluasi dilakukan menggunakan metrik **Accuracy, Precision, Recall, dan F1-Score**.

### Ringkasan Hasil Evaluasi

| Model          | Accuracy | Analisis Model                                                                                                               |   |
| -------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------- | - |
| **LSTM**       | 0.72     | Model paling ringan dan cepat, namun performa paling rendah. Cocok untuk sistem dengan keterbatasan resource komputasi.      |   |
| **BERT Base Uncased** | 0.73     | Memiliki pemahaman konteks teks terbaik sehingga performa optimal, tetapi membutuhkan resource komputasi lebih besar.             |   |
| **DistilBERT**  | 0.73     | Memberikan keseimbangan antara performa dan efisiensi. Lebih cepat dari BERT dengan akurasi yang relatif stabil. |   |

---

## 🌐  Instalasi dan Setup 

### 1. Clone Repositori

bash
git clone https://github.com/faradhitaeka06/UAP-Klasifikasi-Sentimen-Opini-Publik-diTwitter

### 2. Instal PDM dan Dependensi

bash
pip install pdm
pdm init
pdm install

### 3. Jalankan Aplikasi

bash
streamlit run app.py


### 4. Akses dashboard 

bash
Akses dashboard di http://localhost:8501.

---