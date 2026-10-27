# Panduan & Inventaris Dataset Tambahan (`datasets/custom_internet_datasets/`)

Folder ini berisi dataset PPG klinis tambahan (frekuensi sampling **125 Hz**) yang otomatis dibaca oleh [`05_prepare_combined_dataset.py`](../../05_prepare_combined_dataset.py) dan [`01_data_preprocessing.py`](../../01_data_preprocessing.py).

## 1. Struktur Sub-Folder yang Sudah Terisi (Total 57 File Dataset Baru)

1. **`AFib/` (17 File Pasien Fibrilasi Atrium - Kelas `1`)**
   - Berisi rekaman PPG 125 Hz dari 17 pasien AFib berbeda pada PhysioNet MIMIC Database (`mimicdb_sub208_afib_125hz.csv`, `sub209`, `sub210`, `sub216`, `sub219`, `sub226`, `sub237`, `sub248`, `sub252`, `sub254`, `sub408`, `sub409`, `sub417`, `sub439`, `sub442`, `sub474`, `sub482`).
2. **`Normal/` (28 File Pasien Irama Sinus Normal - Kelas `0`, Subtipe `0`)**
   - Berisi 22 pasien irama sinus normal dari PhysioNet MIMIC Database (`mimicdb_sub039_normal_125hz.csv` s/d `sub485_normal_125hz.csv`) ditambah 6 bagian rekaman klinis berdurasi 60 menit (`mimic3_sub3500101_part1..6_125hz.csv`).
3. **`Normal_PAC_PVC_NonAF/` (12 File Kontraksi Prematur Ektopik PAC & PVC - Kelas `0`, Subtipe `2` Hard-Negative)**
   - Berisi 10 rekaman pasien PhysioNet MIMIC dengan denyut prematur ektopik PAC/PVC (`sub041`, `sub221`, `sub224`, `sub230`, `sub276`, `sub404`, `sub410`, `sub414`, `sub427`, `sub456`) serta bank gelombang denyut [`ppg_pac.csv`](Normal_PAC_PVC_NonAF/ppg_pac.csv) dan [`ppg_pvc.csv`](Normal_PAC_PVC_NonAF/ppg_pvc.csv).
   - Digunakan sebagai **Hard-Negative (Non-AFib = 0)** agar model Deep Learning (`1D-DSC + TA-GRU`) belajar membedakan AFib dari PAC/PVC berdasarkan morfologi gelombang (*dicrotic notch*, kemiringan anakrotik) dan pola sekuensial 30 detik.

## 2. Cara Menambahkan Dataset Baru Lagi di Kemudian Hari

Cukup letakkan file `.csv` (dengan kolom `PPG`) ke salah satu dari 3 subfolder di atas:
- `AFib/` untuk rekaman pasien Fibrilasi Atrium
- `Normal/` untuk rekaman pasien Normal Sinus Rhythm
- `Normal_PAC_PVC_NonAF/` untuk rekaman pasien dengan kontraksi prematur PAC/PVC (Non-AFib)

Lalu jalankan kembali:
```bash
python 05_prepare_combined_dataset.py
python 06_train_combined.py
python 04_convert_tflite.py
```
