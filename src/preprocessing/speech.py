import os
import numpy as np
import librosa
from scipy.signal import wiener

NUM_MFCC = 40
MAX_LEN = 500
TARGET_SR = 16000
HOP_LENGTH = 512


def apply_vad(y: np.ndarray, sr: int, top_db: int = 20) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size else y


def denoise_audio(y: np.ndarray) -> np.ndarray:
    try:
        return wiener(y)
    except Exception:
        return y


def preprocess_speech_file(
    file_path: str,
    target_sr: int = TARGET_SR,
    n_mfcc: int = NUM_MFCC,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Load and preprocess a single audio file.

    Pipeline: load → pre-emphasis → Wiener denoise → VAD trim →
              MFCC + delta + delta² extraction.

    Returns:
        ndarray of shape (timesteps, n_mfcc, 3)
    """
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.effects.preemphasis(y)
    y = denoise_audio(y)
    y = apply_vad(y, sr)
    if y.size == 0:
        y = np.zeros(int(target_sr * 0.5))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stack = np.stack([mfcc, delta, delta2], axis=-1)
    return mfcc_stack.transpose(1, 0, 2)


def extract_features(file_path: str, sr: int = 44100, n_mfcc: int = NUM_MFCC) -> np.ndarray:
    """Basic MFCC extraction without advanced preprocessing (used for initial dataset loading)."""
    y, sr = librosa.load(file_path, sr=sr)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def load_and_pad_mfcc(
    file_path: str,
    max_len: int = MAX_LEN,
    n_mfcc: int = NUM_MFCC,
) -> np.ndarray | None:
    """Extract MFCCs from a file and pad/truncate to max_len frames.

    Returns ndarray of shape (max_len, n_mfcc), or None if the file exceeds max_len.
    """
    mfccs = extract_features(file_path, n_mfcc=n_mfcc)
    if mfccs.shape[1] > max_len:
        return None
    pad_width = max_len - mfccs.shape[1]
    return np.pad(mfccs.T, pad_width=((0, pad_width), (0, 0)), mode="constant")
