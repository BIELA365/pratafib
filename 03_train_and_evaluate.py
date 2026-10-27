"""
03_train_and_evaluate.py
================================================================================
Entry-point Pelatihan & Evaluasi Offline Model 1D-DSC + TA-GRU (30 Detik @ 125 Hz).
Memanggil pipeline Two-Stage Training (SimCLR Contrastive Pre-Training +
Supervised Fine-Tuning dengan PAC/PVC Hard-Negative & Evaluasi Bernoulli-Gaussian)
pada `06_train_combined.py`.
================================================================================
"""

from importlib import import_module

if __name__ == "__main__":
    trainer = import_module("06_train_combined")
    trainer.main()
