# 05_Model_AI — Deteksi Fibrilasi Atrium (AFib) Berbasis Sinyal PPG Ear-Worn Menggunakan Deep Learning (`1D-DSC + TA-GRU` + `SimCLR`)

Folder ini berisi pipeline pemodelan *offline* dan *edge deployment* untuk Pra-Tugas Akhir:
**"Perancangan Ear-Worn Berbasis PPG untuk Deteksi Fibrilasi Atrium Menggunakan Deep Learning"**  
*(Abiel Ifan Imanuel Hukom — NRP 5023231004, Teknik Biomedik ITS)*

---

## 1. Spesifikasi Metodologi (Sesuai Proposal Pra-TA Bab 2 & Bab 3)

| Komponen | Spesifikasi Teknis |
| :--- | :--- |
| **Sensor & Optik** | **MAX30105** pada lobulus telinga (*ear-worn*), kanal **Green Optic (~537 nm)** |
| **Frekuensi Sampling ($F_s$)** | **125 Hz** |
| **Pra-Pemrosesan** | Filter Butterworth Band-Pass Orde-4 (**0,4 – 8,0 Hz**) + Normalisasi *Z-Score* |
| **Segmentasi Sinyal** | **30 Detik** (**3.750 sampel** @ 125 Hz) |
| **Ekstraksi Fitur Morfologi & Ritme** | [`PPGMorphologyAndRhythmLayer`](02_model_architecture.py): Mengekstraksi 16 kanal morfologi intra-detak (**PPG**, kemiringan anakrotik/katakrotik **VPG = $d\text{PPG}/dt$**, infleksi ***Dicrotic Notch*** **APG = $d^2\text{PPG}/dt^2$**), 112 kanal autokorelasi ritme PPG, dan 32 kanal konsistensi morfologi VPG antar-detak sepanjang 16 sub-jendela waktu |
| **Arsitektur Deep Learning** | **Hybrid 1D Depthwise Separable CNN + Temporal Attention GRU (`1D-DSC + TA-GRU`)** — Total **40.481 parameter** |
| **Strategi Pelatihan (Bab 2.7 Tanpa GAN)** | **Tahap 1:** *Self-Supervised Contrastive Pre-Training* (**SimCLR** dengan *NT-Xent Loss*) menggunakan operasi **`AddNoise`**, **`Drift`**, dan **Bernoulli–Gaussian** ($n[k] = g[k] + b[k]z[k]$, Persamaan 2.11)<br>**Tahap 2:** *Supervised Fine-Tuning* pada dataset berlabel (**AFib** vs **Non-AFib [Normal Sinus + PAC/PVC]**) |
| **Perangkat Edge Target** | **Raspberry Pi Zero 2 W** + Layar **Waveshare 10,1 Inci** |

---

## 2. Dataset Terpadu (`datasets/` — 86 Subjek Pasien, 14.596 Segmen 30 Detik)

Seluruh dataset terpusat di dalam folder [`datasets/`](datasets/):
1. [`datasets/mimic_perform_af_wfdb/`](datasets/mimic_perform_af_wfdb): 35 subjek pasien (19 AFib, 16 Non-AFib).
2. [`datasets/MIMIC-III_Waveform_Database/`](datasets/MIMIC-III_Waveform_Database) & [`datasets/physionet_annotated_extra/`](datasets/physionet_annotated_extra): Rekaman klinis ICU MIMIC-III (`p010045` & `3500101`).
3. [`datasets/custom_internet_datasets/AFib/`](datasets/custom_internet_datasets/AFib): 14 file pasien AFib dari PhysioNet MIMIC Database (125 Hz).
4. [`datasets/custom_internet_datasets/Normal/`](datasets/custom_internet_datasets/Normal): 28 file pasien Irama Sinus Normal (125 Hz).
5. [`datasets/custom_internet_datasets/Normal_PAC_PVC_NonAF/`](datasets/custom_internet_datasets/Normal_PAC_PVC_NonAF): 13 file pasien dengan kontraksi prematur ektopik (**PAC & PVC**) ditambah bank denyut [`ppg_pac.csv`](datasets/custom_internet_datasets/Normal_PAC_PVC_NonAF/ppg_pac.csv) dan [`ppg_pvc.csv`](datasets/custom_internet_datasets/Normal_PAC_PVC_NonAF/ppg_pvc.csv) sebagai **Hard-Negative (`Kelas 0: Non-AFib`)**.

**Distribusi Segmen (`X_combined.npy` = `(14596, 3750, 1)`):**
- **Kelas 0 — Non-AFib (`6.715` segmen / 46,0%):**
  - *Subtipe 0 (Normal Sinus Rhythm)*: **5.021 segmen**
  - *Subtipe 2 (PAC / PVC Ektopik Hard-Negative)*: **1.694 segmen**
- **Kelas 1 — AFib (`7.881` segmen / 54,0%)**

---

## 3. Hasil Evaluasi pada *Held-Out Subject-Wise Test Set*

Evaluasi dilakukan dengan pembagian ketat per-pasien (*Subject-Wise Group Split*: 62 subjek *Train*, 12 subjek *Val*, **12 subjek *Test* [2.458 segmen] yang belum pernah dilihat model**):

| Metrik Evaluasi | Hasil pada *Held-Out Test Set* |
| :--- | :--- |
| **SimCLR Contrastive Loss (Tahap 1)** | Turun dari **2,3330** (Epoch 1) menjadi **0,4097** (Epoch 8) |
| **Accuracy (Akurasi Uji)** | **95,89%** |
| **Sensitivity / Recall (Deteksi AFib)** | **94,00%** (1.222 / 1.300) |
| **Specificity Total (Non-AFib)** | **98,01%** (1.135 / 1.158) |
| ├── **Specificity pada Normal Sinus Murni** | **99,39%** (815 / 820) |
| └── **Specificity pada Kontraksi Prematur PAC / PVC** | **94,67%** (320 / 338 berhasil ditolak sebagai Non-AFib) |
| **Precision** | **98,15%** |
| **F1-Score** | **96,03%** |
| **ROC-AUC** | **0,9929** |
| **Confusion Matrix** | `TN = 1.135`, `FP = 23`, `FN = 78`, `TP = 1.222` |
| **Akurasi Uji Saat Terkena Artefak (`AddNoise` + `Drift` + `Bernoulli-Gaussian`)** | **96,50%** (F1: **96,65%**, ROC-AUC: **0,9936**) |

### Ukuran & Kecepatan Model TFLite ([`models/tflite_benchmark.json`](models/tflite_benchmark.json))
- [`models/model_float32.tflite`](models/model_float32.tflite): **394,7 KB** — Latensi inferensi rata-rata **6,23 ms / window 30 detik**
- [`models/model_int8.tflite`](models/model_int8.tflite): **302,7 KB** — Latensi inferensi rata-rata **6,19 ms / window 30 detik**

---

## 4. Urutan Menjalankan Program

```bash
# 1. Kurasi & integrasi seluruh dataset (30 detik @ 125 Hz, termasuk PAC/PVC)
python 05_prepare_combined_dataset.py

# 2. Pelatihan Two-Stage (Tahap 1 SimCLR Contrastive + Tahap 2 Supervised Fine-Tuning)
python 06_train_combined.py

# 3. Konversi ke TensorFlow Lite (Float32 & INT8 PTQ) untuk Raspberry Pi Zero 2 W
python 04_convert_tflite.py

# 4. Simulasi inferensi real-time (Normal Sinus vs PAC/PVC vs AFib)
python 07_raspberry_pi_realtime_max30105.py --simulate
```
