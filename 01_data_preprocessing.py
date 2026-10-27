"""
01_data_preprocessing.py
================================================================================
Entry-point Preprocessing & Kurasi Multi-Dataset PPG (30 Detik @ 125 Hz).
Memanggil pipeline terpadu `05_prepare_combined_dataset.py` yang mengintegrasikan:
- MIMIC PERform AF WFDB (35 Subjek)
- MIMIC-III Waveform Database & PhysioNet Extra
- Dataset Baru Internet (AFib, Normal Sinus, serta Hard-Negative PAC & PVC)
================================================================================
"""

from importlib import import_module

if __name__ == "__main__":
    prep = import_module("05_prepare_combined_dataset")
    prep.main()
