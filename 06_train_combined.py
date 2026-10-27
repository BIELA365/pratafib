"""
06_train_combined.py
================================================================================
Pipeline Pelatihan Dua Tahap (Two-Stage Offline Modeling) untuk Deteksi AFib
Berbasis Sinyal PPG 30 Detik (3.750 Sampel @ 125 Hz) sesuai Proposal Pra-TA:
  "Perancangan Ear-Worn Berbasis PPG untuk Deteksi Fibrilasi Atrium
   Menggunakan Deep Learning" (Abiel Ifan Imanuel Hukom - 5023231004)

Tahapan Pemodelan (Sesuai Bab 2.6 & Bab 2.7 Tanpa GAN):
1. Subject-Wise Group Split (85 Subjek Pasien Unik):
   - Mencegah kebocoran data antar pasien (Data Leakage Prevention).
   - Memisahkan secara eksplisit subtipe:
     * Subtipe 0: Normal Sinus Rhythm (Kelas 0: Non-AFib)
     * Subtipe 2: Kontraksi Prematur PAC / PVC (Kelas 0: Non-AFib Hard-Negative)
     * Subtipe 1: Fibrilasi Atrium / AFib (Kelas 1: AFib)
2. Tahap 1 — Self-Supervised Contrastive Pre-Training (SimCLR - Wu et al. [37]):
   - Melatih Encoder 1D-DSC + TA-GRU tanpa label (khususnya memanfaatkan data
     Normal & PAC/PVC yang melimpah) menggunakan fungsi rugi NT-Xent.
   - Mengevaluasi operasi augmentasi fisiologis sesuai Bab 2.7:
     * `Drift` (Pergeseran garis dasar frekuensi rendah akibat respirasi/sensor)
     * `AddNoise` + `Bernoulli-Gaussian` Impulsive Motion Artifact (Persamaan 2.11,
       Rong et al. [39]): n[k] = g[k] + b[k]*z[k]
3. Tahap 2 — Supervised Fine-Tuning & Kalibrasi Bias Logit:
   - Melatih klasifikasi biner AFib (1) vs Non-AFib [Normal + PAC/PVC] (0)
     dengan bobot khusus pada sampel Hard-Negative PAC/PVC.
4. Evaluasi Komprehensif pada Held-Out Subject-Wise Test Set:
   - Metrik Utama (Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUC)
   - Evaluasi Spesifik Hard-Negative PAC/PVC (kemampuan membedakan AFib vs PAC/PVC)
   - Evaluasi Ketahanan Artefak (`AddNoise` + `Drift` + `Bernoulli-Gaussian`)
================================================================================
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib import import_module

arch = import_module("02_model_architecture")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==============================================================================
# 1. PEMODELAN NOISE BAB 2.7: BERNOULLI-GAUSSIAN (PERS. 2.11), ADDNOISE, & DRIFT
# ==============================================================================
def apply_bernoulli_gaussian_and_drift_tf(
    x,
    sigma_g=0.06,
    p_bernoulli=0.03,
    sigma_z=0.55,
    drift_amp=0.18,
):
    """
    Implementasi TensorFlow untuk operasi `AddNoise`, `Drift`, dan proses
    Bernoulli-Gaussian sesuai Bab 2.7 Persamaan (2.11) Proposal Pra-TA:
        n[k] = g[k] + b[k] * z[k]
    di mana:
      - g[k] ~ N(0, sigma_g^2) : Derau latar Gaussian stasioner (`AddNoise`)
      - b[k] ~ Bernoulli(p)    : Sekuens gerbang biner kemunculan artefak impulsif
      - z[k] ~ N(0, sigma_z^2) : Derau impulsif beramplitudo tinggi (sigma_z >> sigma_g)
      - Drift                  : Pergeseran baseline sinusoidal frekuensi rendah (0.05-0.35 Hz)
    """
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]

    # 1. Amplitude scaling acak (variasi tekanan kontak sensor ear-clip MAX30105)
    scale = tf.random.uniform((batch_size, 1, 1), 0.85, 1.18)
    x_aug = x * scale

    # 2. Operasi `Drift` (Baseline Wander frekuensi rendah akibat respirasi & gerak kepala)
    t = tf.linspace(0.0, 30.0, seq_len)
    t = tf.reshape(t, (1, seq_len, 1))
    freq = tf.random.uniform((batch_size, 1, 1), 0.06, 0.35)
    phase = tf.random.uniform((batch_size, 1, 1), 0.0, 2.0 * np.pi)
    amp = tf.random.uniform((batch_size, 1, 1), 0.0, drift_amp)
    drift = amp * tf.math.sin(2.0 * np.pi * freq * t + phase)

    # 3. Proses Bernoulli-Gaussian: n[k] = g[k] + b[k] * z[k] (Persamaan 2.11)
    g_k = tf.random.normal((batch_size, seq_len, 1), mean=0.0, stddev=sigma_g)

    # Untuk menyimulasikan burst artefak gerak sporadis (100-200 ms),
    # bangkitkan gerbang Bernoulli lalu perluas dengan MaxPool pendek
    u_rand = tf.random.uniform((batch_size, seq_len, 1), 0.0, 1.0)
    b_raw = tf.cast(u_rand < (p_bernoulli / 8.0), tf.float32)
    b_k = tf.nn.max_pool1d(b_raw, ksize=11, strides=1, padding="SAME")
    z_k = tf.random.normal((batch_size, seq_len, 1), mean=0.0, stddev=sigma_z)

    n_k = g_k + b_k * z_k
    return x_aug + drift + n_k


def apply_bernoulli_gaussian_and_drift_np(
    X_batch,
    sigma_g=0.06,
    p_bernoulli=0.03,
    sigma_z=0.55,
    drift_amp=0.18,
    seed=123,
):
    """Versi NumPy untuk pengujian ketahanan artefak pada Test Set."""
    rng = np.random.RandomState(seed)
    N, L, _ = X_batch.shape
    out = np.empty_like(X_batch, dtype=np.float32)
    t = np.linspace(0.0, 30.0, L, dtype=np.float32)

    for i in range(N):
        sig = X_batch[i, :, 0].copy()
        # Drift
        f_d = rng.uniform(0.06, 0.35)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        a_d = rng.uniform(0.05, drift_amp)
        drift = a_d * np.sin(2.0 * np.pi * f_d * t + phi)
        # Bernoulli-Gaussian n[k] = g[k] + b[k]*z[k]
        g_k = rng.normal(0.0, sigma_g, size=L)
        b_k = (rng.uniform(0.0, 1.0, size=L) < p_bernoulli).astype(np.float32)
        z_k = rng.normal(0.0, sigma_z, size=L)
        sig_noisy = sig + drift + g_k + b_k * z_k
        sig_noisy = (sig_noisy - np.mean(sig_noisy)) / (np.std(sig_noisy) + 1e-8)
        out[i, :, 0] = sig_noisy
    return out


# ==============================================================================
# 2. TAHAP 1: SELF-SUPERVISED CONTRASTIVE LEARNING (SIMCLR - NT-XENT LOSS)
# ==============================================================================
def nt_xent_contrastive_loss(z1, z2, temperature=0.10):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss untuk SimCLR
    sesuai Wu et al. [37] (Bab 2.7) & `part4_SimCLR.ipynb`.
    Memaksa representasi sinyal PPG bersih/ringan (z1) dan sinyal PPG yang
    terdistorsi `AddNoise` + `Drift` + `Bernoulli-Gaussian` (z2) menjadi identik.
    """
    batch_size = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0)  # (2N, proj_dim), sudah ter-L2-normalisasi

    # Matriks kemiripan kosinus (2N x 2N) dibagi temperatur tau
    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature

    # Masking diagonal utama (kemiripan diri sendiri) dengan nilai sangat negatif
    diag_mask = tf.eye(2 * batch_size, dtype=tf.bool)
    sim_matrix = tf.where(diag_mask, -1e9, sim_matrix)

    # Pasangan positif: indeks i berpasangan dengan i + N (dan sebaliknya)
    labels = tf.concat(
        [tf.range(batch_size, 2 * batch_size), tf.range(0, batch_size)], axis=0
    )
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=sim_matrix
    )
    return tf.reduce_mean(loss)


def pretrain_simclr_contrastive(encoder, X_train, subtypes_train, epochs=8, batch_size=128):
    """
    Melakukan Pre-Training Self-Supervised Contrastive Learning (SimCLR)
    pada Encoder 1D-DSC + TA-GRU menggunakan operasi `AddNoise`, `Drift`, dan
    `Bernoulli-Gaussian` (Persamaan 2.11), dengan penekanan pada dataset
    Normal & PAC/PVC yang berlimpah sesuai Bab 2.7.
    """
    print("\n" + "=" * 80)
    print("TAHAP 1: SELF-SUPERVISED CONTRASTIVE PRE-TRAINING (SimCLR - Bab 2.7)")
    print("Operasi Augmentasi: AddNoise + Drift + Bernoulli-Gaussian (n[k] = g[k] + b[k]z[k])")
    print("=" * 80)

    simclr_model = arch.build_simclr_projection_model(encoder, proj_dim=32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.5e-3)

    # Gunakan seluruh data Normal & PAC/PVC (kelas Non-AFib) ditambah sampel pelatihan
    non_af_mask = (subtypes_train == 0) | (subtypes_train == 2)
    X_normal_heavy = np.concatenate([X_train[non_af_mask], X_train], axis=0)

    ds = (
        tf.data.Dataset.from_tensor_slices(X_normal_heavy)
        .shuffle(buffer_size=min(len(X_normal_heavy), 10000), seed=SEED)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    @tf.function
    def train_contrastive_step(x_batch):
        # View 1: Gangguan ringan (AddNoise + Drift ringan)
        v1 = apply_bernoulli_gaussian_and_drift_tf(
            x_batch, sigma_g=0.03, p_bernoulli=0.01, sigma_z=0.25, drift_amp=0.10
        )
        # View 2: Gangguan kuat (AddNoise + Drift + Artefak Impulsif Bernoulli-Gaussian)
        v2 = apply_bernoulli_gaussian_and_drift_tf(
            x_batch, sigma_g=0.08, p_bernoulli=0.04, sigma_z=0.65, drift_amp=0.22
        )
        with tf.GradientTape() as tape:
            z1 = simclr_model(v1, training=True)
            z2 = simclr_model(v2, training=True)
            loss = nt_xent_contrastive_loss(z1, z2, temperature=0.10)
        grads = tape.gradient(loss, simclr_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, simclr_model.trainable_variables))
        return loss

    loss_history = []
    for ep in range(1, epochs + 1):
        ep_losses = []
        for x_b in ds:
            l_val = train_contrastive_step(x_b)
            ep_losses.append(float(l_val.numpy()))
        mean_l = float(np.mean(ep_losses))
        loss_history.append(mean_l)
        print(f"  [SimCLR Pre-Train] Epoch {ep:02d}/{epochs:02d} - NT-Xent Contrastive Loss: {mean_l:.4f}")

    pretrain_path = os.path.join(MODELS_DIR, "encoder_simclr_pretrained.weights.h5")
    encoder.save_weights(pretrain_path)
    print(f"  [OK] Bobot Encoder Contrastive disimpan ke: {pretrain_path}")
    return loss_history


# ==============================================================================
# 3. PEMBAGIAN SUBJECT-WISE TERSTRATIFIKASI (NORMAL, PAC/PVC, AFIB)
# ==============================================================================
def stratified_subject_group_split(X, y, groups, subtypes, seed=SEED):
    """
    Membagi subjek pasien secara ketat (Group-wise) menjadi Train (70%),
    Val (15%), dan Test (15%) tanpa ada satu pun subjek yang bocor antar-set,
    serta memastikan proporsi subjek Normal, PAC/PVC, dan AFib seimbang.
    """
    unique_groups = np.unique(groups)
    subj_Normal, subj_PacPvc, subj_Afib = [], [], []

    for g in unique_groups:
        mask = groups == g
        y_g = y[mask]
        sub_g = subtypes[mask]
        if np.mean(y_g == 1) > 0.5:
            subj_Afib.append(g)
        elif np.mean(sub_g == 2) > 0.5:
            subj_PacPvc.append(g)
        else:
            subj_Normal.append(g)

    rng = np.random.RandomState(seed)
    train_subj, val_subj, test_subj = set(), set(), set()

    for bucket in [subj_Normal, subj_PacPvc, subj_Afib]:
        bucket = list(bucket)
        rng.shuffle(bucket)
        n = len(bucket)
        n_test = max(2, int(round(0.15 * n)))
        n_val = max(2, int(round(0.15 * n)))
        test_subj.update(bucket[:n_test])
        val_subj.update(bucket[n_test : n_test + n_val])
        train_subj.update(bucket[n_test + n_val :])

    train_idx = np.isin(groups, list(train_subj))
    val_idx = np.isin(groups, list(val_subj))
    test_idx = np.isin(groups, list(test_subj))

    # Pastikan 0 overlap subjek
    assert len(train_subj.intersection(test_subj)) == 0
    assert len(val_subj.intersection(test_subj)) == 0
    assert len(train_subj.intersection(val_subj)) == 0

    return train_idx, val_idx, test_idx, (train_subj, val_subj, test_subj)


# ==============================================================================
# 4. PIPELINE UTAMA (STAGE 1 SIMCLR + STAGE 2 SUPERVISED FINE-TUNING)
# ==============================================================================
def main():
    print("=" * 80)
    print("PELATIHAN OFFLINE MODEL 1D-DSC + TA-GRU (30 DETIK @ 125 HZ)")
    print("Morfologi PPG (PPG, VPG, APG/Dicrotic Notch) + SimCLR Contrastive Learning")
    print("=" * 80)

    X = np.load(os.path.join(DATA_DIR, "X_combined.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_combined.npy"))
    groups = np.load(os.path.join(DATA_DIR, "groups_combined.npy"))
    subtypes = np.load(os.path.join(DATA_DIR, "subtypes_combined.npy"))

    train_idx, val_idx, test_idx, (tr_s, va_s, te_s) = stratified_subject_group_split(
        X, y, groups, subtypes, seed=SEED
    )

    X_train, y_train, sub_train = X[train_idx], y[train_idx], subtypes[train_idx]
    X_val, y_val, sub_val = X[val_idx], y[val_idx], subtypes[val_idx]
    X_test, y_test, sub_test = X[test_idx], y[test_idx], subtypes[test_idx]

    print(f"Pembagian Subject-Wise (Total {len(np.unique(groups))} Subjek Pasien):")
    print(
        f"  - Train : {len(X_train):5d} segmen ({len(tr_s):2d} subjek) | "
        f"Normal={np.sum(sub_train==0)}, PAC/PVC={np.sum(sub_train==2)}, AFib={np.sum(sub_train==1)}"
    )
    print(
        f"  - Val   : {len(X_val):5d} segmen ({len(va_s):2d} subjek) | "
        f"Normal={np.sum(sub_val==0)}, PAC/PVC={np.sum(sub_val==2)}, AFib={np.sum(sub_val==1)}"
    )
    print(
        f"  - Test  : {len(X_test):5d} segmen ({len(te_s):2d} subjek) | "
        f"Normal={np.sum(sub_test==0)}, PAC/PVC={np.sum(sub_test==2)}, AFib={np.sum(sub_test==1)}"
    )

    # -------------------------------------------------------------------------
    # TAHAP 1: Self-Supervised Contrastive Pre-Training (SimCLR - Bab 2.7)
    # -------------------------------------------------------------------------
    encoder = arch.build_encoder_backbone(input_shape=(3750, 1))
    simclr_losses = pretrain_simclr_contrastive(
        encoder, X_train, sub_train, epochs=8, batch_size=128
    )

    # -------------------------------------------------------------------------
    # TAHAP 2: Supervised Fine-Tuning (AFib vs Non-AFib [Normal + PAC/PVC])
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TAHAP 2: SUPERVISED FINE-TUNING KLASIFIKASI AFIB VS NON-AFIB (NORMAL + PAC/PVC)")
    print("=" * 80)

    model = arch.build_1d_dsc_ta_gru_model(input_shape=(3750, 1), encoder=encoder)

    # Berikan bobot sampel (sample_weight) agar kelas seimbang dan sampel
    # Hard-Negative PAC/PVC mendapat perhatian ekstra agar tidak salah dideteksi sebagai AFib
    n_pos = max(int(np.sum(y_train == 1)), 1)
    n_neg = max(int(np.sum(y_train == 0)), 1)
    w_afib = len(y_train) / (2.0 * n_pos)
    w_norm = len(y_train) / (2.0 * n_neg)

    sample_weights_train = np.where(y_train == 1, w_afib, w_norm).astype(np.float32)
    # Penekanan seimbang (1.10x) pada sampel Hard-Negative PAC/PVC
    sample_weights_train[sub_train == 2] *= 1.10

    batch_size = 64

    def augment_supervised(x_b, y_b, w_b):
        x_noisy = apply_bernoulli_gaussian_and_drift_tf(
            x_b, sigma_g=0.04, p_bernoulli=0.025, sigma_z=0.45, drift_amp=0.14
        )
        # Terapkan augmentasi dengan probabilitas 75% selama pelatihan
        apply_flag = tf.cast(
            tf.random.uniform((tf.shape(x_b)[0], 1, 1)) < 0.75, tf.float32
        )
        x_out = apply_flag * x_noisy + (1.0 - apply_flag) * x_b
        return x_out, y_b, w_b

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train.astype(np.float32), sample_weights_train))
        .shuffle(len(X_train), seed=SEED)
        .batch(batch_size)
        .map(augment_supervised, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val.astype(np.float32)))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-3),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="sens"),
            tf.keras.metrics.Precision(name="prec"),
        ],
    )

    best_model_path = os.path.join(MODELS_DIR, "best_model_combined.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=7, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=26,
        callbacks=callbacks,
        verbose=2,
    )

    # Muat model terbaik berdasarkan Validation AUC
    model = tf.keras.models.load_model(best_model_path)

    # -------------------------------------------------------------------------
    # KALIBRASI AMBANG BATAS (BIAS LOGIT) PADA VALIDATION SET
    # -------------------------------------------------------------------------
    val_probs = model.predict(X_val, batch_size=128, verbose=0).ravel()
    best_th, best_score = 0.50, -1.0
    for th in np.linspace(0.20, 0.80, 121):
        preds_v = (val_probs >= th).astype(int)
        # Optimalkan rata-rata F1, Sensitivity, Specificity, dan PAC/PVC Specificity
        f1_v = f1_score(y_val, preds_v, zero_division=0)
        sens_v = recall_score(y_val, preds_v, zero_division=0)
        spec_v = recall_score(1 - y_val, 1 - preds_v, zero_division=0)
        pac_mask_v = sub_val == 2
        pac_spec_v = (
            float(np.mean(preds_v[pac_mask_v] == 0)) if np.any(pac_mask_v) else spec_v
        )
        score = 0.35 * f1_v + 0.30 * sens_v + 0.20 * spec_v + 0.15 * pac_spec_v
        if score > best_score:
            best_score = score
            best_th = float(th)

    # Tanamkan pergeseran logit ke bias layer terakhir (`afib_prob`) agar ambang
    # standar 0.50 pada TFLite Raspberry Pi ekuivalen dengan best_th
    logit_shift = float(np.log(best_th / (1.0 - best_th)))
    out_layer = model.get_layer("afib_prob")
    w_out, b_out = out_layer.get_weights()
    out_layer.set_weights([w_out, b_out - logit_shift])

    # Simpan ke best_model_combined.keras & best_model.keras
    model.save(best_model_path)
    model.save(os.path.join(MODELS_DIR, "best_model.keras"))

    # -------------------------------------------------------------------------
    # EVALUASI KOMPREHENSIF PADA HELD-OUT SUBJECT-WISE TEST SET
    # -------------------------------------------------------------------------
    test_probs = model.predict(X_test, batch_size=128, verbose=0).ravel()
    test_preds = (test_probs >= 0.50).astype(int)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    sens = recall_score(y_test, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    spec = tn / (tn + fp + 1e-8)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)

    # Evaluasi Spesifik Sub-Tipe pada Test Set (Normal Murni vs Hard-Negative PAC/PVC)
    norm_mask_test = sub_test == 0
    pac_mask_test = sub_test == 2
    spec_pure_normal = (
        float(np.mean(test_preds[norm_mask_test] == 0))
        if np.any(norm_mask_test)
        else 0.0
    )
    spec_pac_pvc = (
        float(np.mean(test_preds[pac_mask_test] == 0))
        if np.any(pac_mask_test)
        else 0.0
    )

    # Evaluasi Ketahanan Artefak (Bab 2.7: AddNoise + Drift + Bernoulli-Gaussian) pada Test Set
    X_test_noisy = apply_bernoulli_gaussian_and_drift_np(
        X_test, sigma_g=0.06, p_bernoulli=0.03, sigma_z=0.55, drift_amp=0.18
    )
    noisy_probs = model.predict(X_test_noisy, batch_size=128, verbose=0).ravel()
    noisy_preds = (noisy_probs >= 0.50).astype(int)
    noisy_acc = accuracy_score(y_test, noisy_preds)
    noisy_f1 = f1_score(y_test, noisy_preds, zero_division=0)
    noisy_auc = roc_auc_score(y_test, noisy_probs)

    print("\n" + "=" * 80)
    print("HASIL EVALUASI HELD-OUT TEST SET (SUBJECT-WISE SPLIT - 85 SUBJEK)")
    print("=" * 80)
    print(f"  Accuracy               : {acc*100:.2f}%")
    print(f"  Sensitivity (Recall)   : {sens*100:.2f}%")
    print(f"  Specificity (Non-AFib) : {spec*100:.2f}%")
    print(f"    |-- Specificity pada Normal Sinus Murni : {spec_pure_normal*100:.2f}% ({np.sum(test_preds[norm_mask_test]==0)}/{np.sum(norm_mask_test)})")
    print(f"    '-- Specificity pada PAC / PVC Ektopik  : {spec_pac_pvc*100:.2f}% ({np.sum(test_preds[pac_mask_test]==0)}/{np.sum(pac_mask_test)})")
    print(f"  Precision              : {prec*100:.2f}%")
    print(f"  F1-Score               : {f1*100:.2f}%")
    print(f"  ROC-AUC                : {auc:.4f}")
    print(f"  Confusion Matrix       : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-" * 80)
    print("UJI KETAHANAN ARTEFAK GERAK BAB 2.7 (AddNoise + Drift + Bernoulli-Gaussian):")
    print(f"  Noisy Test Accuracy    : {noisy_acc*100:.2f}%")
    print(f"  Noisy Test F1-Score    : {noisy_f1*100:.2f}%")
    print(f"  Noisy Test ROC-AUC     : {noisy_auc:.4f}")
    print("=" * 80)

    # Simpan metrik lengkap ke JSON
    metrics_summary = {
        "window_seconds": 30,
        "sampling_rate_hz": 125,
        "window_samples": 3750,
        "total_segments": int(len(X)),
        "total_subjects": int(len(np.unique(groups))),
        "test_segments": int(len(X_test)),
        "test_subjects": int(len(te_s)),
        "simclr_contrastive_loss_start": float(simclr_losses[0]),
        "simclr_contrastive_loss_end": float(simclr_losses[-1]),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "specificity_pure_normal": float(spec_pure_normal),
        "specificity_pac_pvc_ectopic": float(spec_pac_pvc),
        "precision": float(prec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "noisy_test_accuracy": float(noisy_acc),
        "noisy_test_f1_score": float(noisy_f1),
        "noisy_test_roc_auc": float(noisy_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open(os.path.join(RESULTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Plot grafik pelatihan & evaluasi
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(range(1, len(simclr_losses) + 1), simclr_losses, "o-", color="#6a1b9a", lw=2)
    axes[0].set_title("Stage 1: SimCLR Contrastive Loss\n(AddNoise + Drift + Bernoulli-Gaussian)")
    axes[0].set_xlabel("Pre-Training Epoch")
    axes[0].set_ylabel("NT-Xent Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Acc", lw=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc", lw=2)
    axes[1].set_title("Stage 2: Supervised Fine-Tuning Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cm = np.array([[tn, fp], [fn, tp]])
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Non-AFib\n(Normal+PAC/PVC)", "AFib"])
    axes[2].set_yticklabels(["Non-AFib\n(Normal+PAC/PVC)", "AFib"])
    axes[2].set_xlabel("Prediksi Model")
    axes[2].set_ylabel("Label Sebenarnya")
    axes[2].set_title(f"Test Confusion Matrix (Acc: {acc*100:.2f}%)")
    for r_i in range(2):
        for c_i in range(2):
            axes[2].text(
                c_i, r_i, str(cm[r_i, c_i]), ha="center", va="center", fontsize=13, fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_and_evaluation_30s.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
