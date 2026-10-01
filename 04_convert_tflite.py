"""
================================================================================
 04_convert_tflite.py
 Konversi Model Keras ke TensorFlow Lite INT8 + Export C Header
================================================================================
 Mengkonversi model 1D-TA-DSC yang telah dilatih menjadi:
   1. TFLite Float32 (baseline, untuk verifikasi akurasi)
   2. TFLite INT8 (kuantisasi penuh, untuk deployment ESP32-S3)
   3. C header file (model_data.h) berisi model sebagai array byte
 
 Kuantisasi INT8:
   - Menggunakan representative dataset dari data preprocessing
   - Seluruh operasi (weights + activations) dikuantisasi ke int8
   - Ukuran model berkurang ~4x dibandingkan float32
   - Kompatibel dengan ESP-NN (akselerasi vektor ESP32-S3)
 
 Penggunaan:
   python 04_convert_tflite.py
   
 Output:
   models/model_float32.tflite     — Model TFLite (float32)
   models/model_int8.tflite        — Model TFLite (INT8 quantized)
   models/model_data.h             — C header untuk embedding di firmware
   models/model_data.cc            — C source file (opsional)
================================================================================
"""

import os
import sys
import numpy as np

# Pastikan parent directory bisa diimport
sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
from importlib import import_module
_arch = import_module("02_model_architecture")
TemporalAttention = _arch.TemporalAttention

# ============================================================================
# KONFIGURASI
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Jumlah sample representative untuk kalibrasi kuantisasi
N_REPRESENTATIVE_SAMPLES = 200

# Nama model array di C header
C_MODEL_NAME = "afib_model"


# ============================================================================
# FUNGSI KONVERSI
# ============================================================================

def load_trained_model():
    """Memuat model Keras yang telah dilatih (atau arsitektur saja)."""
    best_path = os.path.join(MODEL_DIR, "best_model.keras")
    untrained_path = os.path.join(MODEL_DIR, "untrained_model.keras")
    custom_objs = {"TemporalAttention": TemporalAttention}
    
    if os.path.exists(best_path):
        print(f"Memuat model terlatih: {best_path}")
        model = tf.keras.models.load_model(best_path, custom_objects=custom_objs)
        return model, True
    elif os.path.exists(untrained_path):
        print(f"Memuat model arsitektur (belum dilatih): {untrained_path}")
        model = tf.keras.models.load_model(untrained_path, custom_objects=custom_objs)
        return model, False
    else:
        print("[ERROR] Tidak ditemukan model Keras.")
        print("Jalankan '03_train.py' terlebih dahulu.")
        sys.exit(1)


def load_representative_data():
    """Memuat data preprocessing untuk kalibrasi kuantisasi INT8."""
    X_path = os.path.join(DATA_DIR, "ppg_segments.npy")
    
    if os.path.exists(X_path):
        X = np.load(X_path)
        # Ambil subset representatif
        n = min(N_REPRESENTATIVE_SAMPLES, len(X))
        indices = np.random.choice(len(X), n, replace=False)
        return X[indices].astype(np.float32)
    else:
        # Jika data tidak tersedia, buat data sintetis
        print("[WARN] Data preprocessing tidak tersedia.")
        print("       Menggunakan data sintetis untuk kalibrasi kuantisasi.")
        print("       Akurasi kuantisasi mungkin suboptimal.")
        return np.random.randn(N_REPRESENTATIVE_SAMPLES, 625, 1).astype(np.float32)


def convert_to_tflite_float32(model):
    """Konversi model Keras ke TFLite Float32 (baseline)."""
    print("\n--- Konversi ke TFLite Float32 ---")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    output_path = os.path.join(MODEL_DIR, "model_float32.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"  Model Float32 disimpan: {output_path}")
    print(f"  Ukuran: {size_kb:.1f} KB")
    
    return tflite_model


def convert_to_tflite_int8(model, representative_data):
    """
    Konversi model Keras ke TFLite INT8 (kuantisasi penuh).
    
    Kuantisasi INT8 memerlukan "representative dataset" untuk menentukan
    rentang dinamis setiap tensor activation. ESP32-S3 dengan ESP-NN
    mendukung akselerasi hardware untuk operasi INT8.
    """
    print("\n--- Konversi ke TFLite INT8 (Full Quantization) ---")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Aktifkan optimisasi: kuantisasi default (dynamic range)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset generator untuk kalibrasi
    def representative_dataset_gen():
        for i in range(len(representative_data)):
            # Setiap yield harus berupa list tensor dengan shape = input model
            yield [representative_data[i:i+1]]
    
    converter.representative_dataset = representative_dataset_gen
    
    # Paksa seluruh operasi menggunakan INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"  [WARN] Full INT8 gagal ({e}), fallback ke hybrid quantization...")
        # Fallback: INT8 weights + Float32 activations
        converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        converter2.representative_dataset = representative_dataset_gen
        tflite_model = converter2.convert()
    
    output_path = os.path.join(MODEL_DIR, "model_int8.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"  Model INT8 disimpan: {output_path}")
    print(f"  Ukuran: {size_kb:.1f} KB")
    
    return tflite_model


def export_c_header(tflite_model, model_name=C_MODEL_NAME):
    """
    Export model TFLite sebagai C header file (.h) dan source file (.cc).
    
    Format output kompatibel dengan TensorFlow Lite Micro untuk
    embedding langsung di firmware ESP32-S3.
    """
    print("\n--- Export C Header File ---")
    
    model_bytes = bytearray(tflite_model)
    
    # ---- Header file (.h) ----
    header_content = f"""/*
 * ==========================================================================
 *  {model_name}_data.h
 *  Auto-generated by 04_convert_tflite.py
 *  Model: 1D-TA-DSC AFib Detector (TFLite INT8)
 * ==========================================================================
 *  JANGAN EDIT FILE INI SECARA MANUAL.
 *  Jalankan kembali 04_convert_tflite.py untuk regenerasi.
 * ==========================================================================
 */

#ifndef {model_name.upper()}_DATA_H_
#define {model_name.upper()}_DATA_H_

#include <stdint.h>
#include <stddef.h>

// Ukuran model dalam bytes
#define {model_name.upper()}_MODEL_LEN {len(model_bytes)}

// Array model TFLite (stored in flash via PROGMEM/const)
extern const unsigned char {model_name}_model_data[];
extern const unsigned int {model_name}_model_data_len;

#endif  // {model_name.upper()}_DATA_H_
"""
    
    header_path = os.path.join(MODEL_DIR, f"{model_name}_data.h")
    with open(header_path, 'w') as f:
        f.write(header_content)
    print(f"  Header: {header_path}")
    
    # ---- Source file (.cc) ----
    # Konversi bytes ke hex string, 12 nilai per baris
    hex_lines = []
    for i in range(0, len(model_bytes), 12):
        chunk = model_bytes[i:i+12]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"  {hex_str},")
    
    hex_array = "\n".join(hex_lines)
    
    source_content = f"""/*
 * ==========================================================================
 *  {model_name}_data.cc
 *  Auto-generated by 04_convert_tflite.py
 *  Model: 1D-TA-DSC AFib Detector (TFLite INT8)
 *  Ukuran: {len(model_bytes)} bytes ({len(model_bytes)/1024:.1f} KB)
 * ==========================================================================
 */

#include "{model_name}_data.h"

// Simpan di flash memory (ESP32: attribute DROM)
__attribute__((aligned(16)))
const unsigned char {model_name}_model_data[] = {{
{hex_array}
}};

const unsigned int {model_name}_model_data_len = {len(model_bytes)};
"""
    
    source_path = os.path.join(MODEL_DIR, f"{model_name}_data.cc")
    with open(source_path, 'w') as f:
        f.write(source_content)
    print(f"  Source: {source_path}")


def verify_tflite_model(tflite_path, test_input=None):
    """
    Verifikasi model TFLite dapat berjalan dan menghasilkan output valid.
    """
    print(f"\n--- Verifikasi: {os.path.basename(tflite_path)} ---")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")
    
    # Test inference
    if test_input is None:
        test_input = np.random.randn(1, 625, 1).astype(np.float32)
    
    # Kuantisasi input jika perlu
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        input_scale = input_details[0]['quantization_parameters']['scales'][0]
        input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
        test_input_q = (test_input / input_scale + input_zp).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], test_input_q)
    else:
        interpreter.set_tensor(input_details[0]['index'], test_input)
    
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dekuantisasi output jika perlu
    output_dtype = output_details[0]['dtype']
    if output_dtype == np.int8:
        output_scale = output_details[0]['quantization_parameters']['scales'][0]
        output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
        output_float = (output.astype(np.float32) - output_zp) * output_scale
        print(f"  Output (dequantized): {output_float}")
    else:
        print(f"  Output: {output}")
    
    print(f"  [PASS] Inferensi berhasil!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    np.random.seed(42)
    
    print("=" * 70)
    print(" Konversi Model 1D-TA-DSC ke TensorFlow Lite + C Header")
    print("=" * 70)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Muat model
    model, is_trained = load_trained_model()
    
    if not is_trained:
        print("\n[INFO] Model belum dilatih. Konversi tetap dilanjutkan")
        print("       untuk memverifikasi pipeline konversi.")
    
    # 2. Muat data representatif
    rep_data = load_representative_data()
    
    # 3. Konversi ke TFLite Float32
    tflite_f32 = convert_to_tflite_float32(model)
    
    # 4. Konversi ke TFLite INT8
    tflite_int8 = convert_to_tflite_int8(model, rep_data)
    
    # 5. Export C header dari model INT8
    export_c_header(tflite_int8)
    
    # 6. Verifikasi kedua model
    verify_tflite_model(os.path.join(MODEL_DIR, "model_float32.tflite"))
    verify_tflite_model(os.path.join(MODEL_DIR, "model_int8.tflite"))
    
    # 7. Ringkasan
    f32_size = len(tflite_f32) / 1024
    int8_size = len(tflite_int8) / 1024
    compression = (1 - int8_size / f32_size) * 100
    
    print("\n" + "=" * 70)
    print(" KONVERSI SELESAI!")
    print("=" * 70)
    print(f"  Model Float32: {f32_size:.1f} KB")
    print(f"  Model INT8:    {int8_size:.1f} KB  ({compression:.0f}% lebih kecil)")
    print(f"  C Header:      models/{C_MODEL_NAME}_data.h")
    print(f"  C Source:       models/{C_MODEL_NAME}_data.cc")
    print()
    print("  Selanjutnya:")
    print("  1. Salin afib_model_data.h dan afib_model_data.cc ke")
    print("     04_Firmware/esp32-s3/main/")
    print("  2. Build firmware: idf.py build")
    print("  3. Flash ke ESP32-S3: idf.py flash monitor")
    print("=" * 70)


if __name__ == "__main__":
    main()
