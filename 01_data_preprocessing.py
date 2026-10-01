"""
================================================================================
 01_data_preprocessing.py
 Pipeline Preprocessing Sinyal PPG dari MIMIC-III Waveform Database (Normal & AFib)
================================================================================
 Membaca file WFDB (.hea + .dat) untuk Normal Sinus Rhythm (NSR) dan
 Atrial Fibrillation (AFib), mengekstrak kanal PLETH (PPG 125 Hz),
 menangani missing values/NaN dengan interpolasi fisiologis, menerapkan
 bandpass filter (0.5 - 8.0 Hz), normalisasi Z-score, dan segmentasi
 window 5 detik (625 sampel @ 125 Hz, overlap 50%).

 Dilengkapi pemodelan noise sesuai dokumen riset (Abiel Ifan Imanuel Hukom):
   - Model Autoregressive (AR) untuk muscle noise & fluktuasi garis dasar
   - Model Bernoulli-Gaussian untuk motion artifact (wearable hybrid ear-worn)

 Penggunaan:
   python 01_data_preprocessing.py [--augment]
   
 Output:
   data/ppg_segments.npy   — array sinyal PPG (N, 625, 1)
   data/ppg_labels.npy     — array label (N,) → 0=Normal, 1=AFib
   plots/dataset_sample_comparison.png — visualisasi komparasi bentuk gelombang
================================================================================
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# KONFIGURASI
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "03_Dataset", "MIMIC-III Waveform Database")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

SAMPLE_RATE = 125        # Hz (sesuai spesifikasi MIMIC-III & sensor MAX30105)
WINDOW_SEC = 5           # Durasi window dalam detik
WINDOW_LEN = SAMPLE_RATE * WINDOW_SEC  # 625 sampel
OVERLAP_RATIO = 0.5      # 50% overlap antar window
STEP_SIZE = int(WINDOW_LEN * (1 - OVERLAP_RATIO))  # 312 sampel

# Parameter Bandpass Filter (rentang fisiologis PPG kardiovaskular)
BP_LOW = 0.5   # Hz
BP_HIGH = 8.0  # Hz
BP_ORDER = 4


# ============================================================================
# FUNGSI PEMROSESAN SINYAL
# ============================================================================

def bandpass_filter(signal, fs=SAMPLE_RATE, lowcut=BP_LOW, highcut=BP_HIGH, order=BP_ORDER):
    """
    Menerapkan Butterworth bandpass filter zero-phase (filtfilt).
    Menghilangkan DC drift / pernapasan lambat (< 0.5 Hz) dan noise frekuensi tinggi (> 8 Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        low = 0.001
    if high >= 1.0:
        high = 0.999
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def zscore_normalize(signal):
    """
    Standardisasi Z-Score: (x - mean) / std.
    Membuat rentang nilai seragam untuk komparasi bentuk morfologi gelombang.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return signal - mean
    return (signal - mean) / std


def clean_and_interpolate_nan(signal):
    """
    Menangani missing value (NaN) pada sinyal mentah WFDB dengan
    interpolasi linier dan boundary padding.
    """
    if not np.isnan(signal).any():
        return signal
    s = pd.Series(signal)
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.bfill().ffill()
    return s.values


def segment_signal(signal, window_len=WINDOW_LEN, step_size=STEP_SIZE):
    """
    Memotong sinyal kontinyu menjadi window-window dengan overlap.
    Setiap window divalidasi agar bukan flatline/artefak ekstrem.
    """
    segments = []
    start = 0
    while start + window_len <= len(signal):
        seg = signal[start:start + window_len]
        # Validasi deviasi standar (bukan flatline)
        if np.std(seg) > 1e-4:
            segments.append(zscore_normalize(seg))
        start += step_size
    return np.array(segments) if segments else np.empty((0, window_len))


# ============================================================================
# PEMODELAN NOISE STATISTIK (Sesuai Proposal Riset TA)
# ============================================================================

def apply_ar_baseline_wander(signal, noise_level=0.12):
    """
    Model Autoregressive (AR) untuk fluktuasi garis dasar (baseline wander)
    akibat respirasi dan pergeseran sensor.
    """
    # Koefisien AR(2) dengan pole di frekuensi sangat rendah
    a = [1.0, -1.82, 0.83]
    white_noise = np.random.normal(0, 1.0, len(signal))
    ar_noise = lfilter([1.0], a, white_noise)
    ar_noise = zscore_normalize(ar_noise) * noise_level
    return zscore_normalize(signal + ar_noise)


def apply_bernoulli_gaussian_motion_artifact(signal, p_event=0.015, scale=0.45):
    """
    Model Bernoulli-Gaussian untuk motion artifact.
    Mensimulasikan sentuhan/gerakan sesaat pada penjepit telinga (ear-hook fixture).
    """
    # Event acak Bernoulli
    mask = np.random.binomial(1, p_event, len(signal))
    # Pelebaran durasi hentakan gerakan (~0.2 detik)
    smooth_mask = uniform_filter1d(mask.astype(float), size=25) > 0.04
    burst_noise = np.random.normal(0, scale, len(signal)) * smooth_mask
    return zscore_normalize(signal + burst_noise)


# ============================================================================
# PEMBACAAN DATASET WFDB
# ============================================================================

def read_wfdb_pleth(record_path):
    """
    Membaca file WFDB dan mengekstrak kanal PLETH (PPG).
    Mendukung format multi-kanal WFDB.
    """
    try:
        import wfdb
        record = wfdb.rdrecord(record_path)
        sig_names = [name.upper().strip() for name in record.sig_name]
        
        pleth_idx = None
        for candidate in ['PLETH', 'PPG', 'PULSE']:
            if candidate in sig_names:
                pleth_idx = sig_names.index(candidate)
                break
                
        if pleth_idx is None:
            return None, None
            
        raw_signal = record.p_signal[:, pleth_idx]
        cleaned_signal = clean_and_interpolate_nan(raw_signal)
        return cleaned_signal, record.fs
        
    except Exception as e:
        print(f"  [WARN] Error membaca {record_path} via wfdb: {e}")
        return None, None


def determine_record_label(record_path):
    """
    Menentukan label kelas berdasarkan folder/ID rekaman:
      0 = Normal Sinus Rhythm
      1 = Atrial Fibrillation (AFib)
    """
    path_lower = record_path.lower()
    base_name = os.path.basename(record_path).lower()
    
    # Kriteria AFib (sesuai matched ICD-9 427.31 atau penamaan folder)
    if 'afib' in path_lower or '3120713' in base_name or 'p010045' in path_lower:
        return 1
    # Default untuk record 3500101 dsb adalah Normal
    return 0


def find_available_records(dataset_dir):
    """
    Mencari seluruh file record WFDB (.hea yang memiliki pasangan .dat).
    """
    records = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.hea') and '_layout' not in f and not f.endswith('n.hea'):
                rec_name = f[:-4]
                dat_path = os.path.join(root, rec_name + '.dat')
                if os.path.exists(dat_path):
                    records.append(os.path.join(root, rec_name))
    return sorted(records)


# ============================================================================
# VISUALISASI DATASET
# ============================================================================

def plot_sample_comparison(normal_sample, afib_sample, normal_noisy, plot_dir):
    """
    Membuat grafik visualisasi komparasi bentuk gelombang Normal vs AFib.
    """
    os.makedirs(plot_dir, exist_ok=True)
    t = np.arange(len(normal_sample)) / SAMPLE_RATE
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t, normal_sample, color='#2563EB', linewidth=1.5, label='Normal Sinus Rhythm (MIMIC-III)')
    axes[0].set_title('Normal Sinus Rhythm (NSR) — Interval Reguler & Gelombang Dikrotik Jelas', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitudo (Z-score)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    
    axes[1].plot(t, afib_sample, color='#DC2626', linewidth=1.5, label='Atrial Fibrillation (MIMIC-III Matched)')
    axes[1].set_title('Atrial Fibrillation (AFib) — Interval Irama Chaotik & Variasi Amplitudo Puncak', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitudo (Z-score)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    axes[2].plot(t, normal_noisy, color='#059669', linewidth=1.5, label='NSR + AR Baseline Wander & Motion Artifact')
    axes[2].set_title('Augmentasi Noise Statistik — Simulasi Pemakaian Ear-worn Device di Lapangan', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Waktu (detik)', fontsize=11)
    axes[2].set_ylabel('Amplitudo (Z-score)')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plot_file = os.path.join(plot_dir, "dataset_sample_comparison.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f" Grafik sampel disimpan di: {os.path.abspath(plot_file)}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocessing Dataset PPG MIMIC-III (Normal & AFib)")
    parser.add_argument("--augment", action="store_true", help="Tambahkan augmentasi noise AR dan Bernoulli-Gaussian")
    args = parser.parse_args()

    print("=" * 75)
    print(" Pipeline Preprocessing Dataset PPG: Normal vs Atrial Fibrillation (AFib)")
    print(" Format Model 1D-TA-DSC (Input: 625 sampel @ 125 Hz)")
    print("=" * 75)

    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Folder dataset tidak ditemukan: {DATASET_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    records = find_available_records(DATASET_DIR)
    print(f"\nDitemukan {len(records)} record WFDB di folder dataset:")
    for r in records:
        lbl = determine_record_label(r)
        lbl_str = "AFib (1)" if lbl == 1 else "Normal (0)"
        print(f"  - {os.path.basename(r)} -> Kelas: {lbl_str}")

    normal_segments_list = []
    afib_segments_list = []

    for rec_path in tqdm(records, desc="Memproses record"):
        rec_name = os.path.basename(rec_path)
        pleth, fs = read_wfdb_pleth(rec_path)
        
        if pleth is None:
            continue
            
        if len(pleth) < WINDOW_LEN:
            continue

        # Resample jika bukan 125 Hz
        if fs != SAMPLE_RATE:
            from scipy.signal import resample
            n_target = int(len(pleth) * SAMPLE_RATE / fs)
            pleth = resample(pleth, n_target)

        # Bandpass filter (0.5 - 8.0 Hz)
        try:
            pleth_filtered = bandpass_filter(pleth, SAMPLE_RATE, BP_LOW, BP_HIGH, BP_ORDER)
        except Exception as e:
            print(f"  [WARN] Filter gagal pada {rec_name}: {e}")
            continue

        # Segmentasi window 5 detik (625 sampel)
        segs = segment_signal(pleth_filtered, WINDOW_LEN, STEP_SIZE)
        if len(segs) == 0:
            continue

        label = determine_record_label(rec_path)
        if label == 1:
            afib_segments_list.append(segs)
        else:
            normal_segments_list.append(segs)

    if len(normal_segments_list) == 0 and len(afib_segments_list) == 0:
        print("[ERROR] Tidak ada segmen yang berhasil diekstrak!")
        sys.exit(1)

    X_norm = np.concatenate(normal_segments_list, axis=0) if normal_segments_list else np.empty((0, WINDOW_LEN))
    X_afib = np.concatenate(afib_segments_list, axis=0) if afib_segments_list else np.empty((0, WINDOW_LEN))

    print(f"\nHasil Ekstraksi Awal:")
    print(f"  - Segmen Normal riil : {len(X_norm)}")
    print(f"  - Segmen AFib riil   : {len(X_afib)}")

    # Jika salah satu kelas tidak ada atau jomplang, terapkan penyeimbangan/augmentasi
    if len(X_afib) == 0:
        print("\n[WARNING] Tidak ada record AFib ditemukan! Silakan unduh dengan:")
        print("  python 01_download_physionet_afib.py")
        sys.exit(1)

    # Penyeimbangan dataset (Balanced 1:1 Sampling)
    target_count = min(len(X_norm), len(X_afib))
    # Batasi seimbang atau gunakan seluruhnya jika seimbang
    if len(X_norm) > target_count:
        np.random.seed(42)
        idx_norm = np.random.choice(len(X_norm), target_count, replace=False)
        X_norm_balanced = X_norm[idx_norm]
    else:
        X_norm_balanced = X_norm

    X_afib_balanced = X_afib[:target_count]

    # Augmentasi jika diminta
    if args.augment:
        print("\nMenerapkan augmentasi noise statistik (AR + Bernoulli-Gaussian)...")
        aug_norm = [apply_ar_baseline_wander(apply_bernoulli_gaussian_motion_artifact(s)) for s in X_norm_balanced]
        aug_afib = [apply_ar_baseline_wander(apply_bernoulli_gaussian_motion_artifact(s)) for s in X_afib_balanced]
        X_norm_final = np.concatenate([X_norm_balanced, np.array(aug_norm)], axis=0)
        X_afib_final = np.concatenate([X_afib_balanced, np.array(aug_afib)], axis=0)
    else:
        X_norm_final = X_norm_balanced
        X_afib_final = X_afib_balanced

    y_norm = np.zeros(len(X_norm_final), dtype=np.int32)
    y_afib = np.ones(len(X_afib_final), dtype=np.int32)

    X = np.concatenate([X_norm_final, X_afib_final], axis=0)
    y = np.concatenate([y_norm, y_afib], axis=0)

    # Shuffle dataset
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Tambah dimensi channel: (N, 625) -> (N, 625, 1)
    X = X[..., np.newaxis]

    # Simpan ke disk
    np.save(os.path.join(OUTPUT_DIR, "ppg_segments.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "ppg_labels.npy"), y)

    # Buat grafik sampel komparasi
    sample_norm = X_norm_balanced[0] if len(X_norm_balanced) > 0 else np.zeros(WINDOW_LEN)
    sample_afib = X_afib_balanced[0] if len(X_afib_balanced) > 0 else np.zeros(WINDOW_LEN)
    sample_noisy = apply_ar_baseline_wander(apply_bernoulli_gaussian_motion_artifact(sample_norm))
    plot_sample_comparison(sample_norm, sample_afib, sample_noisy, PLOT_DIR)

    print("\n" + "=" * 75)
    print(" PREPROCESSING DAN PENYIAPAN DATASET AFIB SELESAI!")
    print(f" Total segmen tersimpan: {X.shape[0]} segmen")
    print(f" Dimensi tensor        : {X.shape} (N_samples, window_len, channels)")
    print(f" Distribusi kelas      : Normal (0) = {np.sum(y == 0)}, AFib (1) = {np.sum(y == 1)}")
    print(f" Keseimbangan kelas    : {np.sum(y==1)/len(y)*100:.1f}% AFib : {np.sum(y==0)/len(y)*100:.1f}% Normal")
    print(f" File tersimpan di     : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"   - ppg_segments.npy  ({os.path.getsize(os.path.join(OUTPUT_DIR, 'ppg_segments.npy')):,} bytes)")
    print(f"   - ppg_labels.npy    ({os.path.getsize(os.path.join(OUTPUT_DIR, 'ppg_labels.npy')):,} bytes)")
    print("=" * 75)
    print(" Anda sekarang dapat menjalankan pelatihan model dengan:")
    print("   python 03_train.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
