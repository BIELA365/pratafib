"""
04_convert_tflite.py
================================================================================
Konversi Model Keras 1D-DSC + TA-GRU (30 Detik @ 125 Hz = 3.750 Sampel)
ke TensorFlow Lite (Float32 & Post-Training Quantization INT8) untuk
Inferensi Real-Time pada Raspberry Pi Zero 2 W + Sensor MAX30105.
================================================================================
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from importlib import import_module

_arch = import_module("02_model_architecture")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_trained_model():
    """Memuat model 1D-DSC + TA-GRU terbaik hasil pelatihan."""
    custom_objs = {
        "PPGMorphologyAndRhythmLayer": _arch.PPGMorphologyAndRhythmLayer,
        "TemporalAttentionLayer": _arch.TemporalAttentionLayer,
        "L2NormalizeLayer": _arch.L2NormalizeLayer,
    }
    for name in ("best_model_combined.keras", "best_model.keras"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            print(f"Memuat model terlatih: {p}")
            return tf.keras.models.load_model(p, custom_objects=custom_objs)
    raise FileNotFoundError("Model belum ditemukan di folder models/.")


def benchmark_tflite(tflite_path, X_sample, y_sample):
    """Mengukur waktu inferensi per segmen 30 detik & akurasi pada TFLite Interpreter."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    latencies = []
    preds = []
    for i in range(len(X_sample)):
        x_in = X_sample[i : i + 1].astype(np.float32)
        t0 = time.perf_counter()
        interpreter.set_tensor(in_det["index"], x_in)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det["index"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        prob = float(out.ravel()[0])
        preds.append(1 if prob >= 0.50 else 0)

    mean_lat = float(np.mean(latencies[1:]))
    p95_lat = float(np.percentile(latencies[1:], 95))
    acc = float(np.mean(np.array(preds) == y_sample) * 100.0)
    return {
        "mean_latency_ms": round(mean_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "sample_accuracy_percent": round(acc, 2),
    }


def main():
    print("=" * 76)
    print("KONVERSI MODEL 1D-DSC + TA-GRU (30s) KE TFLITE (RASPBERRY PI ZERO 2 W)")
    print("=" * 76)

    model = load_trained_model()
    X = np.load(os.path.join(DATA_DIR, "X_combined.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_combined.npy"))

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(200, len(X)), replace=False)
    X_bench, y_bench = X[idx], y[idx]

    # 1. Konversi TFLite Float32
    conv_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = conv_f32.convert()
    f32_path = os.path.join(MODEL_DIR, "model_float32.tflite")
    with open(f32_path, "wb") as f:
        f.write(tflite_f32)

    # 2. Konversi TFLite PTQ Quantized (INT8 Weights + Dynamic Range)
    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_int8 = conv_int8.convert()
    int8_path = os.path.join(MODEL_DIR, "model_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)

    b_f32 = benchmark_tflite(f32_path, X_bench, y_bench)
    b_int8 = benchmark_tflite(int8_path, X_bench, y_bench)

    f32_kb = len(tflite_f32) / 1024.0
    int8_kb = len(tflite_int8) / 1024.0

    summary = {
        "architecture": "1D-DSC + TA-GRU (30s Morphology + Rhythm + SimCLR)",
        "window_samples": 3750,
        "sampling_rate_hz": 125,
        "total_parameters": int(model.count_params()),
        "float32_size_kb": round(f32_kb, 2),
        "int8_ptq_size_kb": round(int8_kb, 2),
        "float32_benchmark": b_f32,
        "int8_ptq_benchmark": b_int8,
    }
    with open(os.path.join(MODEL_DIR, "tflite_benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  [OK] model_float32.tflite : {f32_kb:.1f} KB | Latensi: {b_f32['mean_latency_ms']:.2f} ms")
    print(f"  [OK] model_int8.tflite    : {int8_kb:.1f} KB | Latensi: {b_int8['mean_latency_ms']:.2f} ms")
    print("=" * 76)


if __name__ == "__main__":
    main()
