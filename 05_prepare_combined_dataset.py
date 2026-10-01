"""Build a leakage-safe PPG dataset from the original and MIMIC PERform records."""

import csv
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "..", "03_Dataset")
ORIGINAL_ROOT = os.path.join(DATASET_ROOT, "MIMIC-III Waveform Database")
PERFORM_ROOT = os.path.join(DATASET_ROOT, "mimic_perform_af_wfdb")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

SAMPLE_RATE = 125
WINDOW_SECONDS = 5
WINDOW_LEN = SAMPLE_RATE * WINDOW_SECONDS
# Non-overlapping windows reduce near-duplicate samples across patients/groups.
STEP_SIZE = WINDOW_LEN
BP_LOW = 0.5
BP_HIGH = 8.0
BP_ORDER = 4


def bandpass_filter(signal):
    nyquist = 0.5 * SAMPLE_RATE
    coefficients = butter(
        BP_ORDER,
        [BP_LOW / nyquist, BP_HIGH / nyquist],
        btype="band",
    )
    return filtfilt(*coefficients, signal)


def normalize(signal):
    centered = signal - np.mean(signal)
    standard_deviation = np.std(centered)
    if standard_deviation < 1e-6:
        return centered
    return centered / standard_deviation


def find_records(root, recursive):
    records = []
    if not os.path.isdir(root):
        return records

    if recursive:
        file_iterator = (
            os.path.join(current_root, filename)
            for current_root, _, filenames in os.walk(root)
            for filename in filenames
        )
    else:
        file_iterator = (
            os.path.join(root, filename) for filename in os.listdir(root)
        )

    for header_path in file_iterator:
        if not header_path.lower().endswith(".hea"):
            continue
        filename = os.path.basename(header_path)
        if "_layout" in filename or filename.lower().endswith("n.hea"):
            continue
        record_path = header_path[:-4]
        if os.path.exists(record_path + ".dat"):
            records.append(record_path)
    return sorted(set(records))


def record_label(record_path, source):
    name = os.path.basename(record_path).lower()
    path = record_path.lower()
    if source == "mimic_perform":
        if name.startswith("mimic_perform_af_"):
            return 1
        if name.startswith("mimic_perform_non_af_"):
            return 0
        raise ValueError(f"Unknown MIMIC PERform record label: {record_path}")
    if "afib" in path or "3120713" in name or "p010045" in path:
        return 1
    return 0


def subject_id_from_record(record, record_path):
    comments = " ".join(getattr(record, "comments", []) or [])
    match = re.search(r"Original Subject ID[^:]*:\s*([^\s;]+)", comments)
    if match:
        return match.group(1)
    return os.path.basename(record_path).split("_")[0]


def read_ppg_record(record_path):
    import wfdb

    record = wfdb.rdrecord(record_path)
    names = [name.split(",", 1)[0].upper().strip() for name in record.sig_name]
    channel_index = next(
        (names.index(candidate) for candidate in ("PLETH", "PPG", "PULSE") if candidate in names),
        None,
    )
    if channel_index is None:
        return None
    signal = record.p_signal[:, channel_index].astype(np.float64)
    signal = pd.Series(signal).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    if record.fs != SAMPLE_RATE:
        target_length = int(round(len(signal) * SAMPLE_RATE / record.fs))
        signal = resample(signal, target_length)
    return signal, subject_id_from_record(record, record_path), record.fs


def segment_signal(signal):
    filtered = bandpass_filter(signal)
    segments = []
    for start in range(0, len(filtered) - WINDOW_LEN + 1, STEP_SIZE):
        segment = filtered[start:start + WINDOW_LEN]
        if np.std(segment) > 1e-4:
            segments.append(normalize(segment).astype(np.float32))
    return segments


def main():
    print("Building combined leakage-safe PPG dataset")
    original_records = find_records(ORIGINAL_ROOT, recursive=True)
    perform_records = find_records(PERFORM_ROOT, recursive=False)
    records = [(path, "mimic_original") for path in original_records]
    records.extend((path, "mimic_perform") for path in perform_records)

    if not records:
        print("[ERROR] No WFDB records were found.")
        sys.exit(1)

    print(f"Records found: {len(records)}")
    print(f"  Original MIMIC: {len(original_records)}")
    print(f"  MIMIC PERform:  {len(perform_records)}")

    segments = []
    labels = []
    groups = []
    metadata = []
    record_summary = {0: 0, 1: 0}

    for record_path, source in tqdm(records, desc="Processing WFDB records"):
        try:
            result = read_ppg_record(record_path)
            if result is None:
                print(f"[WARN] No PPG channel: {record_path}")
                continue
            signal, subject_id, original_fs = result
            label = record_label(record_path, source)
            record_segments = segment_signal(signal)
            if not record_segments:
                print(f"[WARN] No valid windows: {record_path}")
                continue

            record_id = os.path.basename(record_path)
            group_id = f"{source}:{subject_id}"
            for window_index, segment in enumerate(record_segments):
                segments.append(segment)
                labels.append(label)
                groups.append(group_id)
                metadata.append({
                    "window_index": window_index,
                    "record_id": record_id,
                    "subject_id": subject_id,
                    "source": source,
                    "label": label,
                    "original_sampling_rate": float(original_fs),
                })
            record_summary[label] += 1
        except Exception as error:
            print(f"[WARN] Failed {record_path}: {error}")

    if not segments or len(set(labels)) < 2:
        print("[ERROR] Need valid records from both classes.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X = np.expand_dims(np.asarray(segments, dtype=np.float32), axis=-1)
    y = np.asarray(labels, dtype=np.int32)
    group_array = np.asarray(groups)
    order = np.random.default_rng(42).permutation(len(y))
    X, y, group_array = X[order], y[order], group_array[order]

    np.save(os.path.join(OUTPUT_DIR, "combined_ppg_segments.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "combined_ppg_labels.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "combined_ppg_groups.npy"), group_array)

    metadata_path = os.path.join(OUTPUT_DIR, "combined_ppg_metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print("\nCombined dataset ready")
    print(f"  X shape: {X.shape}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Unique groups: {len(np.unique(group_array))}")
    print(f"  Valid records: Normal={record_summary[0]}, AFib={record_summary[1]}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
