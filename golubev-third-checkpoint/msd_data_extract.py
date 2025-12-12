import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os


def extract_segmented_features(filepath, num_parts=8):
    features = {}

    with h5py.File(filepath, "r") as h5:
        if "metadata/songs" in h5:
            meta = h5["metadata/songs"][0]

        if "metadata/artist_terms" in h5:
            artist_terms = h5["metadata/artist_terms"][:]
            if artist_terms.size > 0:
                terms_set = set()
                for term in artist_terms:
                    if isinstance(term, bytes):
                        term = term.decode("utf-8", errors="ignore")
                    terms_set.add(term)
                features["artist_terms"] = "|".join(list(terms_set)[:10])

        artist_name = ""
        title = ""

        if "artist_name" in meta.dtype.names:
            artist_name = meta["artist_name"]
            if isinstance(artist_name, bytes):
                artist_name = artist_name.decode("utf-8", errors="ignore")

        if "title" in meta.dtype.names:
            title = meta["title"]
            if isinstance(title, bytes):
                title = title.decode("utf-8", errors="ignore")

        features["artist_title"] = f"{artist_name} - {title}" if artist_name or title else ""

        if "analysis/songs" in h5:
            analysis = h5["analysis/songs"][0]
            if "duration" in analysis.dtype.names and analysis["duration"].size:
                features["duration"] = float(analysis["duration"])
            if "loudness" in analysis.dtype.names and analysis["loudness"].size:
                features["loudness"] = float(analysis["loudness"])

        if "analysis/segments_timbre" in h5:
            timbre = h5["analysis/segments_timbre"][:]
            if timbre.size > 0:
                features.update(process_array_parts(timbre, "timbre", num_parts))

        if "analysis/segments_pitches" in h5:
            pitches = h5["analysis/segments_pitches"][:]
            if pitches.size > 0:
                features.update(process_array_parts(pitches, "pitches", num_parts))

        return features


def process_array_parts(data, feature_name, num_parts=8):
    features = {}
    N = len(data)

    if N == 0:
        return features

    split_points = np.linspace(0, N, num_parts + 1, dtype=int)

    for part_idx in range(num_parts):
        start_idx = split_points[part_idx]
        end_idx = split_points[part_idx + 1]

        if start_idx < end_idx:
            part_data = data[start_idx:end_idx]
            add_part_features(features, part_data, feature_name, part_idx)
        else:
            add_empty_part_features(features, feature_name, part_idx)

    add_global_features(features, data, feature_name)

    return features


def add_part_features(features, part_data, feature_name, part_idx):
    mean_vals = np.mean(part_data, axis=0)
    for i in range(12):
        features[f"{feature_name}_part{part_idx}_mean_{i}"] = float(mean_vals[i])

    min_vals = np.min(part_data, axis=0)
    for i in range(12):
        features[f"{feature_name}_part{part_idx}_min_{i}"] = float(min_vals[i])

    max_vals = np.max(part_data, axis=0)
    for i in range(12):
        features[f"{feature_name}_part{part_idx}_max_{i}"] = float(max_vals[i])

    range_vals = max_vals - min_vals
    for i in range(12):
        features[f"{feature_name}_part{part_idx}_range_{i}"] = float(range_vals[i])


def add_empty_part_features(features, feature_name, part_idx):
    for i in range(12):
        features[f"{feature_name}_part{part_idx}_mean_{i}"] = 0.0
        features[f"{feature_name}_part{part_idx}_min_{i}"] = 0.0
        features[f"{feature_name}_part{part_idx}_max_{i}"] = 0.0
        features[f"{feature_name}_part{part_idx}_range_{i}"] = 0.0


def add_global_features(features, data, feature_name):
    global_mean = np.mean(data, axis=0)
    for i in range(12):
        features[f"{feature_name}_global_mean_{i}"] = float(global_mean[i])

    global_std = np.std(data, axis=0)
    for i in range(12):
        features[f"{feature_name}_global_std_{i}"] = float(global_std[i])

    global_median = np.median(data, axis=0)
    for i in range(12):
        features[f"{feature_name}_global_median_{i}"] = float(global_median[i])


def build_dataset(data_dir, num_parts=8, output_file="msd_parts_features.csv"):
    files = []
    for ext in ["*.h5", "*.hdf5"]:
        files.extend(list(Path(data_dir).rglob(ext)))


    all_features = []
    for filepath in tqdm(files, desc="Обработка файлов"):
        features = extract_segmented_features(str(filepath), num_parts)
        if features:
            all_features.append(features)

    df = pd.DataFrame(all_features)
    df = df.fillna(0)

    print(f"Размер DataFrame: {df.shape}")

    df.to_csv(output_file, index=False)
    print(f"Сохранено в CSV: {output_file}")

    df_check = pd.read_csv(output_file)
    print(f"Проверка CSV: {df_check.shape}, {df.shape}")


# Основной скрипт
if __name__ == "__main__":
    DATA_DIR = "C://Users//egorg//Downloads//millionsongsubset"
    NUM_PARTS = 8

    build_dataset(
        data_dir=DATA_DIR,
        num_parts=NUM_PARTS,
        output_file="msd_8parts_features.csv"
    )
