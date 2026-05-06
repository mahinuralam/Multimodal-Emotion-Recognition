import os
import numpy as np
import pandas as pd

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

SPEECH_EMOTION_DIRS = {
    "Angry": 0,
    "Calm": 1,
    "Neutral": 2,
    "Happy": 3,
    "Sad": 4,
}

SPEECH_TO_GENERAL = {
    0: 2,  # Angry   -> Negative
    1: 1,  # Calm    -> Positive
    2: 0,  # Neutral -> Neutral
    3: 1,  # Happy   -> Positive
    4: 2,  # Sad     -> Negative
}


def load_speech_dataset(
    dataset_root: str,
    max_len: int = 500,
    num_mfcc: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load speech WAV files and extract padded MFCC features.

    Args:
        dataset_root: Path containing sub-directories named after emotions
                      (Angry, Calm, Neutral, Happy, Sad).
        max_len:      Fixed number of time frames per sample.
        num_mfcc:     Number of MFCC coefficients to extract.

    Returns:
        X:          Feature array of shape (N, max_len, num_mfcc).
        y:          Integer class labels of shape (N,).
        file_paths: Corresponding file paths of shape (N,).
    """
    from src.preprocessing.speech import extract_features

    X, y, file_paths = [], [], []
    for emotion, label in SPEECH_EMOTION_DIRS.items():
        emotion_dir = os.path.join(dataset_root, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        for fname in os.listdir(emotion_dir):
            if not fname.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
                continue
            fpath = os.path.join(emotion_dir, fname)
            mfccs = extract_features(fpath, n_mfcc=num_mfcc)
            if mfccs.shape[1] > max_len:
                continue
            pad_width = max_len - mfccs.shape[1]
            padded = np.pad(mfccs.T, ((0, pad_width), (0, 0)), mode="constant")
            X.append(padded)
            y.append(label)
            file_paths.append(fpath)

    return np.array(X), np.array(y), np.array(file_paths)


def load_eeg_dataset(csv_path: str) -> pd.DataFrame:
    """Load the EEG emotions CSV file and return the raw DataFrame."""
    return pd.read_csv(csv_path)
