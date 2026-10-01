"""
================================================================================
 03_train.py
 Script Pelatihan & Evaluasi Model 1D-TA-DSC
================================================================================
 Melatih model pada data PPG yang telah dipreprocessing, dengan:
   - Stratified train/validation/test split
   - Data augmentation (jitter, scaling, time shift)
   - Early stopping & learning rate scheduling
   - Evaluasi lengkap (akurasi, precision, recall, F1, confusion matrix)
   - Menyimpan model terbaik (.keras) untuk konversi ke TFLite
 
 Penggunaan:
   python 03_train.py
   
 Output:
   models/best_model.keras         — Model terbaik (Keras format)
   models/training_history.npy     — Riwayat pelatihan
   plots/training_curves.png       — Grafik loss & accuracy
   plots/confusion_matrix.png      — Confusion matrix evaluasi
================================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interaktif untuk server/headless
import matplotlib.pyplot as plt

# Pastikan parent directory bisa diimport
sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    ConfusionMatrixDisplay, roc_auc_score
)

# Import arsitektur model
from importlib import import_module
_arch = import_module("02_model_architecture")
build_1d_ta_dsc_model = _arch.build_1d_ta_dsc_model
validate_architecture = _arch.validate_architecture


# ============================================================================
# KONFIGURASI PELATIHAN
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# Hyperparameter
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Label
CLASS_NAMES = ["Normal", "AFib"]


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_signal(signal, noise_std=0.02, scale_range=(0.9, 1.1), 
                   shift_max=10):
    """
    Augmentasi sinyal PPG untuk meningkatkan generalisasi model.
    
    Teknik:
      1. Gaussian jitter — simulasi noise sensor/kuantisasi ADC
      2. Random scaling — variasi amplitudo antar subjek
      3. Random time shift — variasi posisi window relatif thd puncak
    
    Args:
        signal: array (window_len, 1) sinyal PPG
        noise_std: standar deviasi noise Gaussian
        scale_range: rentang faktor skala (min, max)
        shift_max: geser maksimal (sampel)
    
    Returns:
        augmented: sinyal yang telah diaugmentasi
    """
    aug = signal.copy()
    
    # 1. Gaussian jitter
    if np.random.random() > 0.5:
        noise = np.random.normal(0, noise_std, aug.shape)
        aug = aug + noise
    
    # 2. Random amplitude scaling
    if np.random.random() > 0.5:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        aug = aug * scale
    
    # 3. Random circular time shift
    if np.random.random() > 0.5:
        shift = np.random.randint(-shift_max, shift_max + 1)
        aug = np.roll(aug, shift, axis=0)
    
    return aug


class PPGDataGenerator(keras.utils.Sequence):
    """
    Generator data dengan augmentasi on-the-fly untuk pelatihan.
    Menghemat memori dengan tidak menyimpan data augmented secara penuh.
    """
    
    def __init__(self, X, y, batch_size=32, augment=True, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:
                                     (idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]
        
        if self.augment:
            for i in range(len(X_batch)):
                X_batch[i] = augment_signal(X_batch[i])
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# FUNGSI PELATIHAN
# ============================================================================

def load_data():
    """Memuat data preprocessing dari file .npy"""
    X_path = os.path.join(DATA_DIR, "ppg_segments.npy")
    y_path = os.path.join(DATA_DIR, "ppg_labels.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("[ERROR] Data preprocessing belum tersedia.")
        print("Jalankan '01_data_preprocessing.py' terlebih dahulu.")
        sys.exit(1)
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"Data dimuat: X={X.shape}, y={y.shape}")
    print(f"Distribusi label: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


def split_data(X, y):
    """
    Membagi data menjadi train/validation/test set.
    Menggunakan stratified split untuk menjaga proporsi kelas.
    """
    # Pertama: pisahkan test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    
    # Kedua: pisahkan train dan validation
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, 
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"\nPembagian data:")
    print(f"  Train:      {X_train.shape[0]} sampel")
    print(f"  Validation: {X_val.shape[0]} sampel")
    print(f"  Test:       {X_test.shape[0]} sampel")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y_train):
    """
    Menghitung class weight untuk menangani imbalance dataset.
    AFib biasanya jauh lebih sedikit daripada Normal.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = {}
    for cls, count in zip(classes, counts):
        weights[int(cls)] = total / (len(classes) * count)
    
    print(f"Class weights: {weights}")
    return weights


def train_model(X_train, X_val, y_train, y_val, class_weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Melatih model 1D-TA-DSC dengan callback dan monitoring.
    """
    # Buat model
    model = build_1d_ta_dsc_model()
    validate_architecture(model)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    callbacks = [
        # Simpan model terbaik berdasarkan validation accuracy
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping jika tidak ada peningkatan
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Kurangi learning rate saat plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # Data generator dengan augmentasi
    train_gen = PPGDataGenerator(X_train, y_train, batch_size, augment=True)
    val_gen = PPGDataGenerator(X_val, y_val, batch_size, augment=False, 
                                shuffle=False)
    
    # Latih
    print("\n" + "=" * 70)
    print(" MULAI PELATIHAN MODEL 1D-TA-DSC")
    print(f" Total Epochs: {epochs}, Batch Size: {batch_size}")
    print("=" * 70)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Simpan history
    np.save(os.path.join(MODEL_DIR, "training_history.npy"), 
            history.history)
    
    return model, history


# ============================================================================
# EVALUASI & VISUALISASI
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluasi model pada test set dan cetak metrik lengkap."""
    print("\n" + "=" * 70)
    print(" EVALUASI MODEL PADA TEST SET")
    print("=" * 70)
    
    # Prediksi
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                    target_names=CLASS_NAMES, 
                                    digits=4)
    print("\nClassification Report:")
    print(report)
    
    # AUC-ROC (jika ada kedua kelas)
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        print(f"AUC-ROC Score: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return y_pred, y_pred_prob


def plot_training_curves(history):
    """Visualisasi kurva loss dan accuracy selama pelatihan."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGrafik pelatihan disimpan: {save_path}")


def plot_confusion_matrix(y_test, y_pred):
    """Visualisasi confusion matrix."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix — 1D-TA-DSC AFib Detector', 
                 fontsize=14, fontweight='bold')
    
    save_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix disimpan: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pelatihan Model 1D-TA-DSC untuk Deteksi AFib")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah epoch pelatihan (default: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Ukuran batch (default: 32)")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    print("=" * 70)
    print(" Pipeline Pelatihan Model 1D-TA-DSC — Deteksi Atrial Fibrillation")
    print(f" Konfigurasi: {args.epochs} Epochs, Batch Size {args.batch_size}")
    print("=" * 70)
    
    # 1. Muat data
    X, y = load_data()
    
    # Cek minimum data
    n_classes = len(np.unique(y))
    if n_classes < 2:
        print("\n" + "=" * 70)
        print(" ⚠️  PERINGATAN: Dataset hanya memiliki 1 kelas!")
        print("=" * 70)
        print(" Untuk melatih model klasifikasi AFib, dibutuhkan minimal")
        print(" 2 kelas (Normal + AFib).")
        return
    
    # 2. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 3. Hitung class weights
    class_weights = compute_class_weights(y_train)
    
    # 4. Latih model
    model, history = train_model(X_train, X_val, y_train, y_val, class_weights, 
                                 epochs=args.epochs, batch_size=args.batch_size)
    
    # 5. Evaluasi
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    # 6. Visualisasi
    plot_training_curves(history)
    plot_confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print(" PELATIHAN SELESAI!")
    print(f" Model terbaik: {os.path.join(MODEL_DIR, 'best_model.keras')}")
    print(f" Selanjutnya jalankan: python 04_convert_tflite.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
