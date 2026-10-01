"""
================================================================================
 01_download_physionet_afib.py
 Pengunduh Dataset MIMIC-III Atrial Fibrillation (AFib) dari PhysioNet
================================================================================
 Mengunduh rekaman sinyal PPG (PLETH) dari pasien ICU yang terdiagnosis
 Atrial Fibrillation (ICD-9: 427.31) pada MIMIC-III Waveform Database Matched Subset.
 
 Rekaman yang diunduh:
   - Pasien: p010045 (Diagnosis: ICD-9 42731 - Atrial Fibrillation)
   - Record: 3120713_0001 (Durasi: ~42 menit, 316.500 sampel @ 125 Hz)
   - Kanal: PLETH, RESP, II, V, AVR
   
 Penggunaan:
   python 01_download_physionet_afib.py
================================================================================
"""

import os
import sys
import urllib.request

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "03_Dataset", "MIMIC-III Waveform Database")
TARGET_DIR = os.path.join(DATASET_ROOT, "AFib_p010045")
BASE_URL = "https://physionet.org/files/mimic3wdb-matched/1.0/p01/p010045/"

FILES_TO_DOWNLOAD = [
    "3120713_layout.hea",
    "3120713_0001.hea",
    "3120713_0001.dat"
]


def download_file(url, target_path):
    """Mengunduh file dengan progress bar sederhana."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 65536
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                f.write(buffer)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r   [{percent:5.1f}%] {mb_down:.2f} MB / {mb_tot:.2f} MB")
                    sys.stdout.flush()
    print()


def main():
    print("=" * 70)
    print(" Pengunduhan Dataset AFib PPG dari PhysioNet (MIMIC-III Matched)")
    print("=" * 70)
    print(f"Target folder: {os.path.abspath(TARGET_DIR)}")
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    for fn in FILES_TO_DOWNLOAD:
        out_path = os.path.join(TARGET_DIR, fn)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[OK] File sudah ada: {fn} ({os.path.getsize(out_path):,} bytes)")
            continue
        
        url = BASE_URL + fn
        print(f"Mengunduh {fn} dari {url}...")
        try:
            download_file(url, out_path)
            print(f"   Selesai: {fn} ({os.path.getsize(out_path):,} bytes)")
        except Exception as e:
            print(f"   [ERROR] Gagal mengunduh {fn}: {e}")
            sys.exit(1)
            
    print("=" * 70)
    print(" Seluruh file dataset AFib berhasil disiapkan!")
    print(" Jalankan 'python 01_data_preprocessing.py' untuk memproses seluruh dataset.")
    print("=" * 70)


if __name__ == "__main__":
    main()
