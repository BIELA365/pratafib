# Pipeline Pelatihan Model 1D-TA-DSC — Deteksi AFib via PPG

Pipeline Python untuk melatih model deep learning **1D Temporal Attention Depthwise Separable Convolution** dan mengkonversinya ke format TFLite INT8 untuk deployment pada mikrokontroler **ESP32-S3**.

## Prasyarat

- Python 3.11+
- pip

## Instalasi

```bash
cd 05_Model_AI
pip install -r requirements.txt
```

## Pipeline Eksekusi (Berurutan)

### 1. Data Preprocessing
```bash
python 01_data_preprocessing.py
```
Membaca sinyal PPG (kanal PLETH) dari dataset MIMIC-III, menerapkan bandpass filter 0.5–8 Hz, normalisasi Z-score, dan segmentasi menjadi window 5 detik (625 sampel @ 125 Hz).

**Output:** `data/ppg_segments.npy`, `data/ppg_labels.npy`

### 2. Validasi Arsitektur Model
```bash
python 02_model_architecture.py
```
Membangun dan menampilkan arsitektur 1D-TA-DSC. Memvalidasi bahwa jumlah parameter < 50.000 dan estimasi ukuran TFLite INT8 < 100 KB.

### 3. Pelatihan Model
```bash
python 03_train.py
```
Melatih model dengan data augmentation, early stopping, dan learning rate scheduling. Mengevaluasi performa pada test set (akurasi, F1, AUC-ROC, confusion matrix).

**Output:** `models/best_model.keras`, `plots/training_curves.png`, `plots/confusion_matrix.png`

### 4. Konversi ke TFLite + C Header
```bash
python 04_convert_tflite.py
```
Mengkonversi model ke TFLite INT8 (kuantisasi penuh) dan mengekspor sebagai C header file untuk embedding di firmware ESP32-S3.

**Output:** `models/model_int8.tflite`, `models/afib_model_data.h`, `models/afib_model_data.cc`

## Arsitektur Model

```
Input(625,1) → Conv1D(16,k7) → Pool → DSC(32,k5) → Pool
             → DSC(64,k5) → Pool → TemporalAttention(64)
             → DSC(64,k3) → GlobalAvgPool → Dense(32) → Dense(2,softmax)
```

- **Parameter**: ~35.000–45.000
- **TFLite INT8**: ~40–50 KB
- **Target Hardware**: ESP32-S3 (512 KB SRAM + Vector Extension)

## Struktur Folder

```
05_Model_AI/
├── 01_data_preprocessing.py      # Pipeline baca & segmentasi sinyal PPG
├── 02_model_architecture.py      # Definisi arsitektur 1D-TA-DSC (Keras)
├── 03_train.py                   # Pelatihan & evaluasi
├── 04_convert_tflite.py          # Konversi TFLite INT8 + C header
├── requirements.txt              # Dependensi Python
├── README.md                     # Dokumentasi ini
├── data/                         # [auto-generated] data preprocessed
├── models/                       # [auto-generated] model tersimpan
└── plots/                        # [auto-generated] visualisasi hasil
```
