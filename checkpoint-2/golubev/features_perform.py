import librosa
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def extract_features(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path)
        features = {}

        filename = os.path.basename(audio_path)
        track_id = filename.replace(".mp3", "")
        features["track_id"] = track_id
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo_main"] = float(tempo)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6)
        for i in range(6):
            features[f"mfcc_{i}"] = float(np.mean(mfccs[i]))
        features["tembrales_diff_mean"] = float(np.mean([np.std(mfccs[i]) for i in range(6)]))
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = [float(np.mean(chroma[i])) for i in range(12)]
        for i in range(12):
            features[f"chroma_{i}"] = chroma_means[i]
        
        rms = librosa.feature.rms(y=y)
        features["loudness"] = float(np.mean(rms))
        features["energy_variance"] = float(np.std(rms))
        features["average_frequency"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features["zero_crossing"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        print(f"Поймали {track_id}")
        return features
    except Exception:
        print(f"Не поймали {track_id}")
        return None


def process_batch(audio_files):
    with Pool(processes=cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap(extract_features, audio_files), total=len(audio_files)))
    return results


if __name__ == "__main__":
    print("стартунули final")
    audio_folder = "C://Users//egorg//hse-ai-year-project-2025//downloaded_tracks//audio"
    csv_output = "C://Users//egorg//hse-ai-year-project-2025//audio_features_final.csv"
    
    file_names = os.listdir(audio_folder)
    mp3_files = [os.path.join(audio_folder, fname) for fname in file_names if fname.endswith(".mp3")]

    batch_size = 70
    
    for i in range(0, len(mp3_files), batch_size):
        batch = mp3_files[i:i + batch_size]
        
        batch_results = process_batch(batch)
        print(f"сделали батч {i}")
        successful_results = [r for r in batch_results if r is not None]
        
        if successful_results:
            df_batch = pd.DataFrame(successful_results)
            
            if os.path.exists(csv_output):
                df_batch.to_csv(csv_output, mode="a", header=False, index=False, quoting=1)
            else:
                df_batch.to_csv(csv_output, index=False, quoting=1)