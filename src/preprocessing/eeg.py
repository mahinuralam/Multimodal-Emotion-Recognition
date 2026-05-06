import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import zscore
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

LABEL_ENCODING = {"NEUTRAL": 0, "POSITIVE": 1, "NEGATIVE": 2}
FS = 128
WINDOW_SIZE = 128   # 1-second window at 128 Hz
STEP = 64           # 50% overlap
N_ICA_COMPONENTS = 8

# Emotion-specific frequency bands (paper, Section 3)
# Neutral: 5–25 Hz | Negative: broadband 0.5–45 Hz | Positive: high-freq 25–45 Hz
FREQ_BANDS = {
    "neutral":  (5.0,  25.0),
    "negative": (0.5,  45.0),
    "positive": (25.0, 45.0),
}


def butter_bandpass(
    lowcut: float, highcut: float, fs: float, order: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = FS,
    order: int = 5,
) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)


def run_fast_ica(signal: np.ndarray, n_components: int = N_ICA_COMPONENTS) -> np.ndarray:
    """Remove artifacts (muscle movement, eye blinks) via FastICA."""
    length = signal.shape[0]
    pad = (n_components - (length % n_components)) % n_components
    if pad > 0:
        signal = np.pad(signal, (0, pad), mode="edge")
    reshaped = signal.reshape(-1, n_components)
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    try:
        sources = ica.fit_transform(reshaped)
        reconstructed = ica.inverse_transform(sources)
        cleaned = reconstructed.reshape(-1)[:length]
    except Exception:
        cleaned = signal[:length]
    return cleaned


def extract_psd_band_features(window: np.ndarray, fs: float = FS) -> np.ndarray:
    """Compute mean Power Spectral Density in each emotion-specific frequency band.

    Bands (paper, Section 3):
        neutral:  5–25 Hz
        negative: 0.5–45 Hz (broadband)
        positive: 25–45 Hz  (high-frequency)

    Returns:
        ndarray of shape (3,) — one mean PSD value per band
    """
    nperseg = min(len(window), WINDOW_SIZE)
    freqs, psd = welch(window, fs=fs, nperseg=nperseg)
    features = []
    for low, high in FREQ_BANDS.values():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        features.append(np.mean(psd[idx]) if len(idx) > 0 else 0.0)
    return np.array(features, dtype=np.float32)


def preprocess_eeg_sample(
    raw_vector: np.ndarray,
    fs: float = FS,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
) -> np.ndarray:
    """Full EEG preprocessing pipeline for a single sample (paper, Section 3).

    Pipeline:
        1. Band-pass filter (0.5–45 Hz)
        2. FastICA — artifact removal (muscle movements, eye blinks)
        3. Z-score normalisation (zero mean, unit variance)
        4. Segment into overlapping 1-second windows (50% overlap)
        5. Extract PSD features in emotion-specific frequency bands

    Returns:
        ndarray of shape (time_frames, 3) — one PSD feature per band per window
    """
    filtered = bandpass_filter(raw_vector, 0.5, 45.0, fs=fs)
    cleaned = run_fast_ica(filtered)
    normalized = zscore(cleaned)

    if normalized.size < window_size:
        normalized = np.pad(normalized, (0, window_size - normalized.size), mode="edge")

    windows = sliding_window_view(normalized, window_shape=window_size)[::step]
    features = np.stack([extract_psd_band_features(w, fs=fs) for w in windows], axis=0)
    return features  # (time_frames, 3)


def Transform_data(
    EEGData: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode labels, scale features, and one-hot encode targets.

    Returns:
        X_scaled: standardised feature matrix
        Y:        one-hot encoded labels (3 classes)
        X_raw:    raw (unscaled) feature matrix as float32
    """
    data_encoded = EEGData.replace(LABEL_ENCODING)
    x = data_encoded.drop(["label"], axis=1)
    y = data_encoded["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    X_raw = x.values.astype(np.float32)
    Y = to_categorical(y, num_classes=len(LABEL_ENCODING))
    return X_scaled, Y, X_raw
