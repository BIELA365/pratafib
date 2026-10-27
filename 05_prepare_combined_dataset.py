"""
05_prepare_combined_dataset.py
================================================================================
Pipeline Integrasi & Kurasi Multi-Dataset PPG (30 Detik @ 125 Hz = 3.750 Sampel)
Termasuk Dataset Normal Sinus, AFib, serta Hard-Negative PAC & PVC (Kontraksi Prematur)
untuk Deteksi Fibrilasi Atrium Menggunakan Deep Learning (1D-DSC + TA-GRU).

Sumber Dataset Terintegrasi di `05_Model_AI/datasets/`:
1. `datasets/mimic_perform_af_wfdb/` (35 Subjek: 19 AFib, 16 Non-AFib)
2. `datasets/MIMIC-III_Waveform_Database/` & `datasets/physionet_annotated_extra/`
   - Rekaman klinis ICU (AFib p010045 & Normal/Ektopik 3500101 62 menit)
3. `datasets/custom_internet_datasets/AFib/`
   - 17 Pasien AFib baru dari PhysioNet MIMIC Database (125 Hz)
4. `datasets/custom_internet_datasets/Normal/`
   - 22 Pasien Normal Sinus Rhythm baru dari PhysioNet MIMIC + 6 bagian 3500101
5. `datasets/custom_internet_datasets/Normal_PAC_PVC_NonAF/`
   - 10 Pasien Irama Sinus dengan Kontraksi Prematur Ektopik (PAC & PVC)
   - Dataset segmen denyut `ppg_pac.csv` (PAC) & `ppg_pvc.csv` (PVC)
   - Seluruh segmen PAC/PVC diberi label biner `0` (Non-AFib) dan sub-label `2`
     (PAC/PVC Hard-Negative) agar model 1D-DSC + TA-GRU belajar membedakan
     AFib dari PAC/PVC berdasarkan morfologi gelombang (dicrotic notch,
     kemiringan sistolik, amplitudo) dan pola sekuensial 30 detik.

Spesifikasi sesuai Proposal Pra-TA (Bab 2 & Bab 3):
- Frekuensi Sampling (Fs) : 125 Hz
- Band-Pass Filter        : Butterworth Orde 4, 0.4 - 8.0 Hz (Zero-Phase filtfilt)
- Durasi Segmentasi       : 30 Detik (3.750 sampel per segmen)
- Normalisasi             : Z-Score per segmen 30 detik
================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FS = 125
WINDOW_SEC = 30
WINDOW_SAMPLES = TARGET_FS * WINDOW_SEC  # 3.750 sampel (30 detik @ 125 Hz)
LOWCUT_HZ = 0.4
HIGHCUT_HZ = 8.0
FILTER_ORDER = 4


def bandpass_filter_ppg(sig, fs=TARGET_FS):
    """Filter Butterworth Band-Pass 0.4 - 8.0 Hz sesuai Proposal Bab 3."""
    nyq = 0.5 * fs
    b, a = butter(FILTER_ORDER, [LOWCUT_HZ / nyq, HIGHCUT_HZ / nyq], btype="band")
    return filtfilt(b, a, sig)


def zscore_normalize(seg):
    """Normalisasi Z-Score per segmen 30 detik."""
    std = np.std(seg)
    if std < 1e-6:
        return None
    return ((seg - np.mean(seg)) / (std + 1e-8)).astype(np.float32)


def compute_morphology_and_rhythm_sqi(seg_z, fs=TARGET_FS):
    """
    Mengekstraksi indikator kualitas sinyal (SQI) dan karakteristik fisiologis
    tanpa membuang segmen PAC/PVC (kontraksi prematur).
    Mengembalikan: (is_valid_sensor, cv_ibi, median_sub_cv, notch_cv)
    """
    if np.any(np.isnan(seg_z)) or np.any(np.isinf(seg_z)):
        return False, 0.0, 0.0, 0.0
    if np.max(np.abs(seg_z)) > 12.0:
        return False, 0.0, 0.0, 0.0

    peaks, _ = find_peaks(seg_z, distance=int(0.30 * fs), prominence=0.32)
    # Dalam 30 detik, detak jantung manusia wajar berada di rentang 15 - 95 detak (30 - 190 BPM)
    if len(peaks) < 15 or len(peaks) > 95:
        return False, 0.0, 0.0, 0.0

    ibi = np.diff(peaks) / float(fs)
    mean_ibi = np.mean(ibi)
    if mean_ibi <= 0:
        return False, 0.0, 0.0, 0.0

    cv_ibi = np.std(ibi) / (mean_ibi + 1e-8)

    # Hitung CV IBI pada sub-jendela 6 detik (untuk membedakan PAC/PVC sporadis vs AFib persisten)
    sub_cvs = []
    for k in range(5):
        s_lo, s_hi = k * 750, (k + 1) * 750
        p_sub = peaks[(peaks >= s_lo) & (peaks < s_hi)]
        if len(p_sub) >= 3:
            ibi_sub = np.diff(p_sub) / float(fs)
            sub_cvs.append(np.std(ibi_sub) / (np.mean(ibi_sub) + 1e-8))
    median_sub_cv = float(np.median(sub_cvs)) if sub_cvs else cv_ibi

    # Variasi morfologi Dicrotic Notch dari turunan kedua (APG = d^2PPG/dt^2) pada fase katakrotik
    apg = np.diff(seg_z, n=2)
    cat_energies = [
        np.std(apg[p + 8 : min(p + 34, len(apg))])
        for p in peaks[:-1]
        if p + 34 < len(apg)
    ]
    notch_cv = (
        float(np.std(cat_energies) / (np.mean(cat_energies) + 1e-8))
        if len(cat_energies) >= 5
        else 0.5
    )

    return True, float(cv_ibi), median_sub_cv, notch_cv


def extract_windows_from_signal(
    raw_sig,
    fs,
    binary_label,
    subject_id,
    stride_samples,
    subtype_override=None,
):
    """
    Memfilter sinyal, memotong menjadi segmen 30 detik (3.750 sampel),
    dan mengklasifikasikan sub-tipe fisiologis:
      - subtype 0 : Normal Sinus Rhythm (y = 0)
      - subtype 1 : Atrial Fibrillation / AFib (y = 1)
      - subtype 2 : Normal Sinus + PAC/PVC Ectopic Contractions (y = 0, Hard-Negative)
    """
    sig = pd.Series(np.asarray(raw_sig, dtype=np.float64)).interpolate().bfill().ffill().values
    if len(sig) < int(WINDOW_SEC * fs):
        return [], [], [], []

    if int(round(fs)) != TARGET_FS:
        target_len = int(round(len(sig) * TARGET_FS / float(fs)))
        sig = resample(sig, target_len)

    if np.std(sig) < 1e-5:
        return [], [], [], []

    filtered = bandpass_filter_ppg(sig, fs=TARGET_FS)

    X_list, y_list, g_list, sub_list = [], [], [], []
    for start in range(0, len(filtered) - WINDOW_SAMPLES + 1, stride_samples):
        seg = filtered[start : start + WINDOW_SAMPLES]
        seg_z = zscore_normalize(seg)
        if seg_z is None:
            continue

        valid, cv_ibi, median_sub_cv, notch_cv = compute_morphology_and_rhythm_sqi(seg_z)
        if not valid:
            continue

        if binary_label == 1:
            # Untuk rekaman pasien AFib: pastikan segmen benar-benar berada dalam episode fibrilasi
            # persisten (bukan episode sinus normal / paroksismal tenang di tengah rekaman ICU)
            if cv_ibi < 0.10 or median_sub_cv < 0.085:
                continue
            subtype = 1
        else:
            # Untuk rekaman Non-AFib (Normal Sinus maupun PAC/PVC):
            if subtype_override == 2:
                # Rekaman yang mengandung denyut ektopik PAC/PVC:
                # Mayoritas sub-jendela tetap irama sinus teratur (median_sub_cv <= 0.085)
                if median_sub_cv > 0.085 or cv_ibi > 0.16:
                    continue
                subtype = 2
            else:
                # Rekaman Non-AFib umum:
                if cv_ibi <= 0.075 and median_sub_cv <= 0.065:
                    subtype = 0
                elif cv_ibi <= 0.145 and median_sub_cv <= 0.070:
                    subtype = 2  # PAC/PVC sporadis alami di dalam rekaman Non-AFib
                else:
                    continue

        X_list.append(seg_z[:, np.newaxis])
        y_list.append(int(binary_label))
        g_list.append(str(subject_id))
        sub_list.append(int(subtype))

    return X_list, y_list, g_list, sub_list


def generate_coupled_pac_pvc_windows(clean_normal_windows, pac_csv_path, pvc_csv_path):
    """
    Membangun segmen 30 detik Hard-Negative PAC & PVC realistis dengan menyisipkan
    morfologi denyut ektopik asli dari `ppg_pac.csv` dan `ppg_pvc.csv`
    (beserta karakteristik fisiologis: coupling interval memendek, amplitudo sistolik
    mengecil / hilangnya dicrotic notch pada detak ektopik, diikuti compensatory pause
    dan post-extrasystolic potentiation) ke dalam segmen irama sinus normal.
    Semua segmen ini berlabel biner y = 0 (Non-AFib) dan subtype = 2 (PAC/PVC).
    """
    rng = np.random.RandomState(42)
    pac_beats, pvc_beats = [], []

    for path, target_list in [(pac_csv_path, pac_beats), (pvc_csv_path, pvc_beats)]:
        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
            for idx in range(len(df)):
                row = pd.to_numeric(df.iloc[idx, 1:], errors="coerce").dropna().values
                if len(row) >= 400:
                    # Downsample dari 500 Hz ke 125 Hz (faktor 4)
                    row_125 = row[::4]
                    row_125 = bandpass_filter_ppg(row_125, fs=TARGET_FS)
                    row_z = zscore_normalize(row_125)
                    if row_z is not None:
                        target_list.append(row_z)

    X_ect, y_ect, g_ect, sub_ect = [], [], [], []
    if not clean_normal_windows:
        return X_ect, y_ect, g_ect, sub_ect

    # Pilih sampel jendela normal untuk disisipi 2 s/d 4 kejadian PAC atau PVC dalam 30 detik
    num_to_make = min(len(clean_normal_windows), 450)
    indices = rng.choice(len(clean_normal_windows), size=num_to_make, replace=False)

    for idx_i, win_idx in enumerate(indices):
        base_win, subj_id = clean_normal_windows[win_idx]
        sig = base_win[:, 0].copy()

        # Sisipkan 1 hingga 2 kejadian ektopik terisolasi (PAC atau PVC) di dalam 30 detik
        n_events = rng.randint(1, 3)
        centers = sorted(rng.choice(range(600, WINDOW_SAMPLES - 600, 900), size=n_events, replace=False))
        is_pvc = (idx_i % 2 == 1)
        beat_bank = pvc_beats if (is_pvc and len(pvc_beats) > 0) else pac_beats

        for c in centers:
            if len(beat_bank) > 0:
                ect_template = beat_bank[rng.randint(0, len(beat_bank))]
                L = min(len(ect_template), 85)  # ~0.68 detik (satu siklus denyut prematur + jeda kompensasi)
                alpha = np.hanning(L)
                ect_segment = ect_template[:L].copy()
                ect_segment[: L // 2] *= rng.uniform(0.50, 0.72)
                ect_segment[L // 2 : (3 * L) // 4] *= 0.25  # Jeda kompensasi (diastolic pause)
                ect_segment[(3 * L) // 4 :] *= rng.uniform(1.08, 1.22)  # Post-extrasystolic potentiation
                sig[c : c + L] = (1.0 - alpha) * sig[c : c + L] + alpha * ect_segment

        sig_z = zscore_normalize(sig)
        if sig_z is not None:
            X_ect.append(sig_z[:, np.newaxis])
            y_ect.append(0)  # Non-AFib!
            # Gunakan subject_id yang sama dengan subjek asalnya agar tidak bocor saat GroupShuffleSplit!
            g_ect.append(subj_id)
            sub_ect.append(2)  # Sub-label 2 = PAC/PVC Ectopic

    return X_ect, y_ect, g_ect, sub_ect


def main():
    print("=" * 80)
    print("KURASI & INTEGRASI MULTI-DATASET PPG (30 DETIK @ 125 HZ = 3.750 SAMPEL)")
    print("Termasuk Normal Sinus, AFib, serta Hard-Negative PAC & PVC (Non-AFib)")
    print("=" * 80)

    X_all, y_all, groups_all, subtypes_all = [], [], [], []
    clean_normal_pool = []

    # -------------------------------------------------------------------------
    # 1. Dataset Utama: MIMIC PERform AF WFDB (35 Subjek: 19 AFib, 16 Non-AFib)
    # -------------------------------------------------------------------------
    wfdb_dir = os.path.join(DATASETS_DIR, "mimic_perform_af_wfdb")
    hea_files = sorted(glob.glob(os.path.join(wfdb_dir, "*.hea")))
    print(f"[1/5] Memuat MIMIC PERform AF WFDB ({len(hea_files)} subjek)...")
    for hea in hea_files:
        rec_name = os.path.splitext(os.path.basename(hea))[0]
        if not rec_name.startswith("mimic_perform_"):
            continue
        is_af = "_af_" in rec_name and "_non_af_" not in rec_name
        label = 1 if is_af else 0
        try:
            rec = wfdb.rdrecord(os.path.splitext(hea)[0])
            ch_idx = 0
            for idx, sname in enumerate(rec.sig_name):
                if "PPG" in sname.upper() or "PLETH" in sname.upper():
                    ch_idx = idx
                    break
            raw_ppg = rec.p_signal[:, ch_idx]
            # Stride 375 sampel (3 detik) untuk AFib & Non-AFib
            stride = 375
            Xw, yw, gw, sw = extract_windows_from_signal(
                raw_ppg, rec.fs, label, f"perform_{rec_name}", stride_samples=stride
            )
            X_all.extend(Xw)
            y_all.extend(yw)
            groups_all.extend(gw)
            subtypes_all.extend(sw)
            for x_item, s_item, g_item in zip(Xw, sw, gw):
                if s_item == 0:
                    clean_normal_pool.append((x_item, g_item))
        except Exception as e:
            print(f"  [!] Gagal membaca {rec_name}: {e}")

    # -------------------------------------------------------------------------
    # 2. Dataset MIMIC-III Waveform Database (AFib p010045 & Normal 3500101)
    # -------------------------------------------------------------------------
    print("[2/5] Memuat MIMIC-III Waveform Database & PhysioNet Extra...")
    extra_af_records = [
        os.path.join(DATASETS_DIR, "MIMIC-III_Waveform_Database", "AFib_p010045", "3120713_0001"),
        os.path.join(DATASETS_DIR, "physionet_annotated_extra", "AFib_p010045_extra", "3120713_0002"),
        os.path.join(DATASETS_DIR, "physionet_annotated_extra", "AFib_p010045_extra", "3120713_0005"),
        os.path.join(DATASETS_DIR, "physionet_annotated_extra", "AFib_p010045_extra", "3746698_0002"),
    ]
    for rec_base in extra_af_records:
        if os.path.exists(rec_base + ".hea"):
            try:
                rec = wfdb.rdrecord(rec_base)
                for idx, sname in enumerate(rec.sig_name):
                    if "PLETH" in sname.upper() or "PPG" in sname.upper():
                        subj = "mimic3_p010045_" + os.path.basename(rec_base)
                        Xw, yw, gw, sw = extract_windows_from_signal(
                            rec.p_signal[:, idx], rec.fs, 1, subj, stride_samples=750
                        )
                        X_all.extend(Xw)
                        y_all.extend(yw)
                        groups_all.extend(gw)
                        subtypes_all.extend(sw)
                        break
            except Exception as e:
                print(f"  [!] Skip {rec_base}: {e}")

    # -------------------------------------------------------------------------
    # 3. Dataset Baru Internet: AFib (17 Pasien PhysioNet MIMIC Database)
    # -------------------------------------------------------------------------
    custom_af_dir = os.path.join(DATASETS_DIR, "custom_internet_datasets", "AFib")
    af_csvs = sorted(glob.glob(os.path.join(custom_af_dir, "*.csv")))
    print(f"[3/5] Memuat Dataset Baru AFib ({len(af_csvs)} file pasien)...")
    for csv_path in af_csvs:
        fname = os.path.splitext(os.path.basename(csv_path))[0]
        subj_id = "_".join(fname.split("_")[:2])  # e.g., mimicdb_sub208
        try:
            df = pd.read_csv(csv_path)
            col = "PPG" if "PPG" in df.columns else df.columns[0]
            # Gunakan stride 125 sampel (1 detik) agar pasien baru terwakili optimal
            Xw, yw, gw, sw = extract_windows_from_signal(
                df[col].values, TARGET_FS, 1, subj_id, stride_samples=125
            )
            X_all.extend(Xw)
            y_all.extend(yw)
            groups_all.extend(gw)
            subtypes_all.extend(sw)
        except Exception as e:
            print(f"  [!] Gagal membaca {fname}: {e}")

    # -------------------------------------------------------------------------
    # 4. Dataset Baru Internet: Normal Sinus Rhythm (28 File Pasien Baru)
    # -------------------------------------------------------------------------
    custom_norm_dir = os.path.join(DATASETS_DIR, "custom_internet_datasets", "Normal")
    norm_csvs = sorted(glob.glob(os.path.join(custom_norm_dir, "*.csv")))
    print(f"[4/5] Memuat Dataset Baru Normal Sinus ({len(norm_csvs)} file pasien)...")
    for csv_path in norm_csvs:
        fname = os.path.splitext(os.path.basename(csv_path))[0]
        subj_id = "_".join(fname.split("_")[:2])  # e.g., mimicdb_sub039 atau mimic3_sub3500101
        stride = 500 if "3500101" in fname else 125
        try:
            df = pd.read_csv(csv_path)
            col = "PPG" if "PPG" in df.columns else df.columns[0]
            Xw, yw, gw, sw = extract_windows_from_signal(
                df[col].values, TARGET_FS, 0, subj_id, stride_samples=stride
            )
            X_all.extend(Xw)
            y_all.extend(yw)
            groups_all.extend(gw)
            subtypes_all.extend(sw)
            for x_item, s_item, g_item in zip(Xw, sw, gw):
                if s_item == 0:
                    clean_normal_pool.append((x_item, g_item))
        except Exception as e:
            print(f"  [!] Gagal membaca {fname}: {e}")

    # -------------------------------------------------------------------------
    # 5. Dataset Baru Hard-Negative: PAC & PVC (Kontraksi Prematur Non-AFib)
    # -------------------------------------------------------------------------
    custom_pac_dir = os.path.join(DATASETS_DIR, "custom_internet_datasets", "Normal_PAC_PVC_NonAF")
    pac_csvs = sorted(glob.glob(os.path.join(custom_pac_dir, "mimicdb_*.csv")))
    print(f"[5/5] Memuat Dataset Hard-Negative PAC & PVC ({len(pac_csvs)} pasien + bank denyut PAC/PVC)...")
    for csv_path in pac_csvs:
        fname = os.path.splitext(os.path.basename(csv_path))[0]
        subj_id = "_".join(fname.split("_")[:2])  # e.g., mimicdb_sub041
        try:
            df = pd.read_csv(csv_path)
            col = "PPG" if "PPG" in df.columns else df.columns[0]
            Xw, yw, gw, sw = extract_windows_from_signal(
                df[col].values,
                TARGET_FS,
                0,
                subj_id,
                stride_samples=125,
                subtype_override=2,
            )
            X_all.extend(Xw)
            y_all.extend(yw)
            groups_all.extend(gw)
            subtypes_all.extend(sw)
        except Exception as e:
            print(f"  [!] Gagal membaca {fname}: {e}")

    # Tambahkan juga segmen PAC/PVC terkopel dari `ppg_pac.csv` dan `ppg_pvc.csv`
    pac_bank_path = os.path.join(custom_pac_dir, "ppg_pac.csv")
    pvc_bank_path = os.path.join(custom_pac_dir, "ppg_pvc.csv")
    X_ect, y_ect, g_ect, sub_ect = generate_coupled_pac_pvc_windows(
        clean_normal_pool, pac_bank_path, pvc_bank_path
    )
    X_all.extend(X_ect)
    y_all.extend(y_ect)
    groups_all.extend(g_ect)
    subtypes_all.extend(sub_ect)

    # Konversi ke NumPy Array
    X_arr = np.asarray(X_all, dtype=np.float32)
    y_arr = np.asarray(y_all, dtype=np.int32)
    groups_arr = np.asarray(groups_all)
    subtypes_arr = np.asarray(subtypes_all, dtype=np.int32)

    # Simpan ke folder `data/`
    np.save(os.path.join(OUTPUT_DIR, "X_combined.npy"), X_arr)
    np.save(os.path.join(OUTPUT_DIR, "y_combined.npy"), y_arr)
    np.save(os.path.join(OUTPUT_DIR, "groups_combined.npy"), groups_arr)
    np.save(os.path.join(OUTPUT_DIR, "subtypes_combined.npy"), subtypes_arr)

    # Sinkronkan juga ke X.npy, y.npy, groups.npy
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X_arr)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_arr)
    np.save(os.path.join(OUTPUT_DIR, "groups.npy"), groups_arr)

    n_total = len(y_arr)
    n_afib = int(np.sum(y_arr == 1))
    n_non_afib = int(np.sum(y_arr == 0))
    n_pure_normal = int(np.sum(subtypes_arr == 0))
    n_pac_pvc = int(np.sum(subtypes_arr == 2))
    n_subjects = len(np.unique(groups_arr))

    print("-" * 80)
    print("RINGKASAN DATASET GABUNGAN (30 DETIK @ 125 HZ):")
    print(f"  Total Segmen (X.shape)             : {X_arr.shape}")
    print(f"  Total Subjek / Grup Pasien Unik    : {n_subjects} subjek")
    print(f"  Kelas 0 - Non-AFib Total           : {n_non_afib} segmen ({100*n_non_afib/n_total:.1f}%)")
    print(f"    |-- Subtipe 0 (Normal Sinus)     : {n_pure_normal} segmen")
    print(f"    '-- Subtipe 2 (PAC / PVC Ektopik): {n_pac_pvc} segmen (Hard-Negative)")
    print(f"  Kelas 1 - AFib (Fibrilasi Atrium)  : {n_afib} segmen ({100*n_afib/n_total:.1f}%)")
    print("-" * 80)


if __name__ == "__main__":
    main()
