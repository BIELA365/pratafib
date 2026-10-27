"""
================================================================================
 03_train.py
 Script Pelatihan Terpadu Model 1D-DSC + TA-GRU (Subject-Disjoint Evaluation)
================================================================================
 Menjalankan pelatihan dan evaluasi model 1D-DSC + TA-GRU menggunakan pipeline
 terpadu yang bebas kebocoran data antar-subjek (subject-disjoint split).

 Penggunaan:
   python 03_train.py [--epochs 30] [--no-augmentation]
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

if __name__ == "__main__":
    train_combined = import_module("06_train_combined")
    train_combined.main()
