"""
================================================================================
 07_raspberry_pi_realtime_max30105.py
 Sistem Real-Time Akuisisi MAX30105 + Inferensi TFLite + Waveshare Display
 (Target: Raspberry Pi Zero 2 W — Sesuai Bab 3 Proposal & Diagram Blok ETS)
================================================================================
 Alur Pemrosesan pada Raspberry Pi Zero 2 W (Gambar 3.1 Proposal / Slide 10 ETS):
   1. Akuisisi Data: Membaca sinyal PPG digital dari sensor MAX30105 (I2C 0x57)
                     pada frekuensi sampling 125 Hz (atau mode --simulate).
   2. Band-pass Filter: Filter Butterworth Orde-4 dengan frekuensi 0,4 - 8,0 Hz.
   3. Segmentasi Window: Membagi sinyal menjadi segmen berdurasi tetap 30 detik
                         (3.750 sampel @ 125 Hz) menggunakan ring buffer.
   4. Normalisasi: Outlier clipping + Z-score normalization per window.
   5. Inferensi Model: Menjalankan model TFLite 1D-DSC + TA-GRU (model_int8.tflite).
   6. Hasil Klasifikasi & Display: Menampilkan gelombang PPG real-time, status
                                   NORMAL / AF TERDETEKSI, HR (BPM), & PRV (RMSSD)
                                   pada layar Waveshare 10,1 inci atau terminal.

 Cara Menjalankan:
   - Mode Simulasi & Uji GUI (di PC atau RPi tanpa sensor terpasang):
       python 07_raspberry_pi_realtime_max30105.py --simulate --gui
   - Mode Sensor MAX30105 Langsung di Raspberry Pi Zero 2 W:
       python 07_raspberry_pi_realtime_max30105.py --gui
================================================================================
"""

import argparse
import collections
import os
import time
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt, savgol_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

SAMPLE_RATE = 125
WINDOW_SECONDS = 30
WINDOW_LEN = SAMPLE_RATE * WINDOW_SECONDS  # 3.750 sampel (30 detik @ 125 Hz)
BP_LOW = 0.4
BP_HIGH = 8.0
BP_ORDER = 4


# ============================================================================
# 1. DRIVER SENSOR MAX30105 (I2C Raspberry Pi Zero 2 W: SDA=GPIO2, SCL=GPIO3)
# ============================================================================

class MAX30105Sensor:
    """
    Driver I2C ringan untuk sensor optik MAX30105 / MAX30102 pada Raspberry Pi Zero 2 W.
    Menggunakan smbus2 pada bus I2C-1 (alamat default 0x57).
    """

    I2C_ADDR = 0x57
    REG_INTR_STATUS_1 = 0x00
    REG_INTR_ENABLE_1 = 0x02
    REG_FIFO_WR_PTR = 0x04
    REG_OVF_COUNTER = 0x05
    REG_FIFO_RD_PTR = 0x06
    REG_FIFO_DATA = 0x07
    REG_FIFO_CONFIG = 0x08
    REG_MODE_CONFIG = 0x09
    REG_SPO2_CONFIG = 0x0A
    REG_LED1_PA = 0x0C  # Red LED
    REG_LED2_PA = 0x0D  # IR LED
    REG_LED3_PA = 0x0E  # Green LED (530-537 nm — optimal untuk ear-worn PPG)
    REG_MULTI_LED_CTRL1 = 0x11

    def __init__(self, bus_num=1, address=I2C_ADDR):
        import smbus2
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        self._setup_sensor()

    def _setup_sensor(self):
        # Reset MAX30105
        self.bus.write_byte_data(self.address, self.REG_MODE_CONFIG, 0x40)
        time.sleep(0.1)
        # Clear FIFO pointers
        self.bus.write_byte_data(self.address, self.REG_FIFO_WR_PTR, 0x00)
        self.bus.write_byte_data(self.address, self.REG_OVF_COUNTER, 0x00)
        self.bus.write_byte_data(self.address, self.REG_FIFO_RD_PTR, 0x00)
        # FIFO Config: sample averaging = 4, FIFO rollover enabled
        self.bus.write_byte_data(self.address, self.REG_FIFO_CONFIG, 0x50)
        # Mode Config: Multi-LED mode (0x07) dengan Slot 1 = Green LED (537 nm)
        self.bus.write_byte_data(self.address, self.REG_MODE_CONFIG, 0x07)
        self.bus.write_byte_data(self.address, self.REG_MULTI_LED_CTRL1, 0x03)
        # SpO2 Config: ADC range 4096nA, Sample Rate ~400 Hz (4x avg -> ~100-125 Hz), Pulse Width 411us (18-bit)
        self.bus.write_byte_data(self.address, self.REG_SPO2_CONFIG, 0x2F)
        # LED Pulse Amplitude (~7-12 mA untuk jaringan lobulus/earlobe, fokus Green LED3)
        self.bus.write_byte_data(self.address, self.REG_LED1_PA, 0x00)
        self.bus.write_byte_data(self.address, self.REG_LED2_PA, 0x00)
        self.bus.write_byte_data(self.address, self.REG_LED3_PA, 0x28)

    def read_sample(self):
        """Membaca 1 sampel PPG 18-bit dari kanal Green Optic (537 nm) FIFO MAX30105."""
        data = self.bus.read_i2c_block_data(self.address, self.REG_FIFO_DATA, 3)
        green = ((data[0] & 0x03) << 16) | (data[1] << 8) | data[2]
        # Sinyal PPG reflektansi diinvert agar puncak sistolik menghadap ke atas
        return -float(green)


# ============================================================================
# 2. PREPROCESSING REAL-TIME (Bab 3 Proposal: Butterworth 0,4-8 Hz & Z-score)
# ============================================================================

def preprocess_30s_window(raw_signal, fs=SAMPLE_RATE):
    """
    Tahap preprocessing sinyal PPG 30 detik sesuai Bab 3 Proposal:
      1. Butterworth band-pass filter 0,4 - 8,0 Hz (Orde-4)
      2. Median + Savitzky-Golay smoothing
      3. Outlier clipping (median +/- 3*sigma) + Z-score normalization
    """
    sig = np.asarray(raw_signal, dtype=np.float64)
    nyq = 0.5 * fs
    b, a = butter(BP_ORDER, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    filtered = filtfilt(b, a, sig)
    smoothed = savgol_filter(medfilt(filtered, kernel_size=5), window_length=21, polyorder=3)

    med = np.median(smoothed)
    sigma = np.std(smoothed)
    if sigma < 1e-6:
        return (smoothed - med).astype(np.float32)
    clipped = np.clip(smoothed, med - 3.0 * sigma, med + 3.0 * sigma)
    mean_c = np.mean(clipped)
    std_c = np.std(clipped)
    if std_c < 1e-6:
        return (clipped - mean_c).astype(np.float32)
    return ((clipped - mean_c) / std_c).astype(np.float32)


def compute_prv_metrics(norm_seg, fs=SAMPLE_RATE):
    """Menghitung Heart Rate (BPM), RMSSD (ms), dan CV-IBI dari puncak sistolik PPG."""
    peaks, _ = find_peaks(norm_seg, distance=int(0.30 * fs), prominence=0.35)
    if len(peaks) < 3:
        return 0.0, 0.0, 0.0
    ibi = np.diff(peaks) / float(fs)
    mean_ibi = float(np.mean(ibi))
    hr_bpm = 60.0 / mean_ibi if mean_ibi > 0 else 0.0
    rmssd_ms = float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000.0) if len(ibi) >= 2 else 0.0
    cv_ibi = float(np.std(ibi) / mean_ibi) if mean_ibi > 0 else 0.0
    return hr_bpm, rmssd_ms, cv_ibi


# ============================================================================
# 3. ENGINE INFERENSI TENSORFLOW LITE (Raspberry Pi Zero 2 W)
# ============================================================================

class AFibTFLiteEngine:
    """Wrapper TFLite Interpreter untuk inferensi lokal di Raspberry Pi Zero 2 W."""

    def __init__(self, model_path=None):
        if model_path is None:
            int8_path = os.path.join(MODEL_DIR, "model_int8.tflite")
            f32_path = os.path.join(MODEL_DIR, "model_float32.tflite")
            model_path = int8_path if os.path.exists(int8_path) else f32_path

        self.model_path = model_path
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.expected_len = int(self.input_details[0]["shape"][1])

    def predict_window(self, norm_segment):
        """Menjalankan inferensi TFLite pada 1 window PPG dan mengembalikan (kelas, prob_afib, latency_ms)."""
        if len(norm_segment) != self.expected_len:
            norm_segment = np.interp(
                np.linspace(0, len(norm_segment) - 1, self.expected_len),
                np.arange(len(norm_segment)),
                norm_segment,
            ).astype(np.float32)

        x_in = norm_segment.reshape(1, self.expected_len, 1).astype(np.float32)
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]["index"], x_in)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"]).ravel()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob_afib = float(out[0]) if len(out) == 1 else float(out[1])
        pred_class = 1 if prob_afib >= 0.50 else 0
        return pred_class, prob_afib, latency_ms


# ============================================================================
# 4. DASHBOARD WAVESHARE DISPLAY 10,1 INCI & MODE TERMINAL
# ============================================================================

def run_realtime_monitor(simulate=False, use_gui=False):
    engine = AFibTFLiteEngine()
    print("=" * 76)
    print(" PPG EAR-WORN AFib MONITORING — RASPBERRY PI ZERO 2 W (GREEN OPTIC 537 nm)")
    print("=" * 76)
    print(f"  Model TFLite : {os.path.basename(engine.model_path)}")
    print(f"  Durasi Window: {engine.expected_len / SAMPLE_RATE:.0f} detik ({engine.expected_len} sampel @ {SAMPLE_RATE} Hz)")
    print(f"  Filter       : Butterworth Band-Pass {BP_LOW}-{BP_HIGH} Hz + Z-Score")
    print(f"  Mode Akuisisi: {'Simulasi Dataset Uji (Normal, PAC/PVC, AFib)' if simulate else 'Sensor I2C MAX30105 Green LED (0x57)'}")
    print("=" * 76)

    sim_segments, sim_subtypes = None, None
    if simulate:
        seg_path = os.path.join(DATA_DIR, "X_combined.npy")
        sub_path = os.path.join(DATA_DIR, "subtypes_combined.npy")
        if os.path.exists(seg_path) and os.path.exists(sub_path):
            sim_segments = np.load(seg_path)
            sim_subtypes = np.load(sub_path)

    if use_gui:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.canvas.manager.set_window_title("Waveshare 10.1 Display - PPG Ear-Worn AFib Monitoring")

    step_idx = 0
    ring_buffer = collections.deque(maxlen=WINDOW_LEN)
    sensor = None if simulate else MAX30105Sensor()

    try:
        while True:
            if simulate and sim_segments is not None:
                # Siklus demonstrasi 3 subtipe: 0=Normal Sinus, 2=PAC/PVC (Non-AFib), 1=AFib
                cycle_subtypes = [0, 2, 1]
                target_sub = cycle_subtypes[step_idx % 3]
                matching = np.where(sim_subtypes == target_sub)[0]
                chosen = matching[(step_idx // 3) % len(matching)]
                norm_seg = sim_segments[chosen, :, 0]
                sub_names = {0: "Normal Sinus", 2: "PAC/PVC Ektopik [Non-AFib]", 1: "AFib"}
                true_str = f" (Ground Truth: {sub_names[target_sub]})"
            else:
                print("Mengakuisisi data sensor MAX30105 (5 detik update)...")
                n_read = WINDOW_LEN if len(ring_buffer) < WINDOW_LEN else SAMPLE_RATE * 5
                for _ in range(n_read):
                    ring_buffer.append(sensor.read_sample())
                    time.sleep(1.0 / SAMPLE_RATE)
                norm_seg = preprocess_30s_window(np.array(ring_buffer))
                true_str = ""

            pred_class, prob_afib, lat_ms = engine.predict_window(norm_seg)
            hr_bpm, rmssd_ms, cv_ibi = compute_prv_metrics(norm_seg)
            status_text = "AF TERDETEKSI (Pola Tidak Teratur)" if pred_class == 1 else "NORMAL (Irama Teratur)"

            print(
                f"[Window #{step_idx+1:02d}] Prediksi: {status_text}{true_str} | "
                f"P(AFib)={prob_afib*100:5.1f}% | HR={hr_bpm:5.1f} BPM | "
                f"RMSSD={rmssd_ms:5.1f} ms | Inferensi={lat_ms:.2f} ms"
            )

            if use_gui:
                ax.clear()
                t_axis = np.arange(len(norm_seg)) / float(SAMPLE_RATE)
                color = "#d62728" if pred_class == 1 else "#2ca02c"
                ax.plot(t_axis, norm_seg, color=color, linewidth=1.2)
                ax.set_title(
                    f"PPG EAR-WORN MONITORING | Status: {status_text}\n"
                    f"Prob AFib: {prob_afib*100:.1f}% | HR: {hr_bpm:.1f} BPM | "
                    f"PRV RMSSD: {rmssd_ms:.1f} ms | TFLite Latency: {lat_ms:.2f} ms",
                    fontsize=11,
                    fontweight="bold",
                    color=color,
                )
                ax.set_xlabel("Waktu (detik)")
                ax.set_ylabel("Amplitudo PPG (Z-Score)")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                plt.pause(2.5 if simulate else 0.05)

            step_idx += 1
            if simulate and step_idx >= 6 and not use_gui:
                break
            if simulate and not use_gui:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nMonitoring dihentikan oleh pengguna.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time PPG AFib Monitoring pada Raspberry Pi Zero 2 W")
    parser.add_argument("--simulate", action="store_true", help="Jalankan simulasi menggunakan sampel dataset")
    parser.add_argument("--gui", action="store_true", help="Tampilkan dashboard grafik untuk layar Waveshare 10.1 inci")
    args = parser.parse_args()
    run_realtime_monitor(simulate=args.simulate, use_gui=args.gui)
