import os
import numpy as np
import librosa
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def get_features(y: np.ndarray, sr: int) -> dict:
    features = {}

    def add_stats(name, mat):
        features[f"{name}_mean"] = float(np.mean(mat))
        features[f"{name}_var"] = float(np.var(mat))

    add_stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr))
    add_stats("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr))
    add_stats("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr))
    add_stats("zero_crossing_rate", librosa.feature.zero_crossing_rate(y=y))
    add_stats("rms", librosa.feature.rms(y=y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, coeff in enumerate(mfcc, start=1):
        add_stats(f"mfcc_{i}", coeff)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, c in enumerate(chroma, start=1):
        add_stats(f"chroma_{i}", c)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i, band in enumerate(contrast, start=1):
        add_stats(f"spectral_contrast_{i}", band)

    add_stats("spectral_flatness", librosa.feature.spectral_flatness(y=y))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i, t in enumerate(tonnetz, start=1):
        add_stats(f"tonnetz_{i}", t)

    return features


def process_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=90)
        feats = get_features(y, sr)
        feats["id"] = os.path.splitext(os.path.basename(file_path))[0]
        return feats
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None


if __name__ == "__main__":
    audio_folder = r"C:\Users\Эдгар\Desktop\downloaded_tracks\audio"
    all_files = [os.path.join(audio_folder, f)
                 for f in os.listdir(audio_folder)
                 if f.endswith((".mp3", ".wav"))]

    rows = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            result = future.result()
            if result is not None:
                rows.append(result)

    df_features = pd.DataFrame(rows)
    print("Извлечено треков:", len(df_features))
    print("Размерность:", df_features.shape)
    print(df_features.head())

    output_path = r"C:\Users\Эдгар\Desktop\downloaded_tracks\audio_features_named.csv"
    df_features.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Файл сохранён: {output_path}")
