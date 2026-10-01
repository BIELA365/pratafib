"""Train 1D-TA-DSC with patient/record-grouped splits and realistic augmentation."""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

_architecture = import_module("02_model_architecture")
build_1d_ta_dsc_model = _architecture.build_1d_ta_dsc_model
validate_architecture = _architecture.validate_architecture

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
CLASS_NAMES = ["Normal", "AFib"]


# The following two functions implement the noise models used only for train batches.
def zscore(signal):
    centered = signal - np.mean(signal)
    standard_deviation = np.std(centered)
    return centered if standard_deviation < 1e-6 else centered / standard_deviation


def apply_ar_baseline_wander(signal, noise_level=0.12):
    coefficients = [1.0, -1.82, 0.83]
    white_noise = np.random.normal(0.0, 1.0, len(signal))
    ar_noise = lfilter([1.0], coefficients, white_noise)
    ar_noise = zscore(ar_noise) * noise_level
    return zscore(signal + ar_noise)


def apply_motion_artifact(signal, event_probability=0.015, scale=0.45):
    event_mask = np.random.binomial(1, event_probability, len(signal))
    smooth_mask = uniform_filter1d(event_mask.astype(float), size=25) > 0.04
    burst_noise = np.random.normal(0.0, scale, len(signal)) * smooth_mask
    return zscore(signal + burst_noise)


def augment_signal(signal):
    augmented = signal.copy()
    if np.random.random() < 0.5:
        augmented += np.random.normal(0.0, 0.02, augmented.shape)
    if np.random.random() < 0.5:
        augmented *= np.random.uniform(0.9, 1.1)
    if np.random.random() < 0.5:
        augmented = np.roll(augmented, np.random.randint(-10, 11), axis=0)
    if np.random.random() < 0.5:
        augmented = apply_ar_baseline_wander(augmented[:, 0])[:, None]
    if np.random.random() < 0.35:
        augmented = apply_motion_artifact(augmented[:, 0])[:, None]
    return augmented.astype(np.float32)


class PPGDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, augment, shuffle):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indices].copy()
        if self.augment:
            for position in range(len(X_batch)):
                X_batch[position] = augment_signal(X_batch[position])
        return X_batch, self.y[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_data():
    X = np.load(os.path.join(DATA_DIR, "combined_ppg_segments.npy"))
    y = np.load(os.path.join(DATA_DIR, "combined_ppg_labels.npy"))
    groups = np.load(os.path.join(DATA_DIR, "combined_ppg_groups.npy"))
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y, and groups must have equal lengths")
    print(f"Data: X={X.shape}, y={y.shape}, groups={len(np.unique(groups))}")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X.astype(np.float32), y.astype(np.int32), groups


def split_by_group(X, y, groups):
    unique_groups, first_indices = np.unique(groups, return_index=True)
    group_labels = y[first_indices]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    group_folds = list(splitter.split(unique_groups, group_labels))
    test_group_names = unique_groups[group_folds[0][1]]
    validation_group_names = unique_groups[group_folds[1][1]]
    test_indices = np.flatnonzero(np.isin(groups, test_group_names))
    validation_indices = np.flatnonzero(np.isin(groups, validation_group_names))
    held_out_indices = np.concatenate([test_indices, validation_indices])
    train_indices = np.setdiff1d(np.arange(len(X)), held_out_indices, assume_unique=False)

    train_group_set = set(groups[train_indices])
    validation_group_set = set(groups[validation_indices])
    test_group_set = set(groups[test_indices])
    if train_group_set & validation_group_set or train_group_set & test_group_set or validation_group_set & test_group_set:
        raise RuntimeError("Group leakage detected in split")

    print("\nGrouped split:")
    for name, indices in (("Train", train_indices), ("Validation", validation_indices), ("Test", test_indices)):
        print(f"  {name}: windows={len(indices)}, groups={len(np.unique(groups[indices]))}, labels={dict(zip(*np.unique(y[indices], return_counts=True)))}")
    return train_indices, validation_indices, test_indices


def class_weights(y):
    labels, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(label): total / (len(labels) * count) for label, count in zip(labels, counts)}


def train_model(X_train, y_train, X_val, y_val, weights, epochs, augment_train):
    model = build_1d_ta_dsc_model()
    validate_architecture(model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "best_model_combined.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    train_generator = PPGDataGenerator(X_train, y_train, BATCH_SIZE, augment=augment_train, shuffle=True)
    validation_generator = PPGDataGenerator(X_val, y_val, BATCH_SIZE, augment=False, shuffle=False)
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )
    np.save(os.path.join(MODEL_DIR, "training_history_combined.npy"), history.history)
    return model, history, model_path


def evaluate(model, X_test, y_test):
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0, output_dict=True)
    metrics = {
        "accuracy": float(np.mean(predictions == y_test)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "auc_roc": float(roc_auc_score(y_test, probabilities[:, 1])) if len(np.unique(y_test)) == 2 else None,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))
    return predictions, metrics


def save_plots(history, y_test, predictions):
    os.makedirs(PLOT_DIR, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Combined grouped training loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set_title("Combined grouped training accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(PLOT_DIR, "training_curves_combined.png"), dpi=150)
    plt.close(figure)

    matrix_figure, matrix_axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=CLASS_NAMES).plot(
        ax=matrix_axis, cmap="Blues", values_format="d"
    )
    matrix_axis.set_title("Combined grouped test confusion matrix")
    matrix_figure.savefig(os.path.join(PLOT_DIR, "confusion_matrix_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(matrix_figure)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    X, y, groups = load_data()
    train_indices, validation_indices, test_indices = split_by_group(X, y, groups)
    weights = class_weights(y[train_indices])
    print(f"Class weights: {weights}")
    model, history, model_path = train_model(
        X[train_indices], y[train_indices],
        X[validation_indices], y[validation_indices],
        weights, args.epochs, augment_train=not args.no_augmentation,
    )
    predictions, metrics = evaluate(model, X[test_indices], y[test_indices])
    with open(os.path.join(MODEL_DIR, "evaluation_combined.json"), "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    np.savez(
        os.path.join(DATA_DIR, "combined_grouped_split_indices.npz"),
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
    )
    save_plots(history, y[test_indices], predictions)
    print(f"\nBest model: {model_path}")


if __name__ == "__main__":
    main()
