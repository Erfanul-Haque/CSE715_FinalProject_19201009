"""Module for extracting audio features (mel-spectrograms and chroma) from .wav files."""
import librosa
import numpy as np
import torch

def extract_audio_features(audio_path, sr=22050):
    """Extracts log-mel spectrograms, chroma features, and beat-synchronous segments.
    
    Args:
        audio_path (str): The file path to the .wav audio file.
        sr (int): The sample rate to load the audio with. Defaults to 22050.
        
    Returns:
        dict: A dictionary containing log-mel spectrograms, chroma, and beat-synced features as PyTorch tensors.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # We use '_' because we don't need the tempo value, only the beat_frames
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    mel_sync = librosa.util.sync(log_mel_spec, beat_frames, aggregate=np.mean)
    
    return {
        'log_mel_spec': torch.tensor(log_mel_spec, dtype=torch.float32),
        'chroma_sync': torch.tensor(chroma_sync, dtype=torch.float32).T,
        'mel_sync': torch.tensor(mel_sync, dtype=torch.float32).T
    }