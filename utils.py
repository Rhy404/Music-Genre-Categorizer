# this file is the Audio Processor
#this code will handle the logic that will handle a raw MP3 file :)

import librosa
import numpy as np
import pandas as pd
import joblib

def extract_audio_features(file_path):
    # Load audio file (30 seconds)
    y, sr = librosa.load(file_path, duration=30)
    
    features = {}

    # 1. Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)

    # 2. RMS (Root Mean Square)
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    # 3. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_var'] = np.var(spec_cent)

    # 4. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_var'] = np.var(spec_bw)

    # 5. Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)

    # 7. Harmony and Percussive (Perceptr)
    harmony, perceptr = librosa.effects.hpss(y)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)
    features['perceptr_mean'] = np.mean(perceptr)
    features['perceptr_var'] = np.var(perceptr)

    # 8. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo[0])

    # 9. MFCCs (1-20)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i-1])

    return pd.DataFrame([features])

print(extract_audio_features("FrEliseWoo59.mp3"))

model = joblib.load('music_genre_xgb.pkl')
le = joblib.load('label_encoder.pkl')

features_df = extract_audio_features("FrEliseWoo59.mp3")

prediction_numeric = model.predict(features_df)
genre_name = le.inverse_transform(prediction_numeric)

print(f"Predicted Genre: {genre_name[0]}")



