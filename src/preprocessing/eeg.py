import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import zscore
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

LABEL_ENCODING = {"NEUTRAL": 0, "POSITIVE": 1, "NEGATIVE": 2}
FS = 128
WINDOW_SIZE = 128
STEP = 64
N_ICA_COMPONENTS = 8


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


def preprocess_eeg_sample(
    raw_vector: np.ndarray,
    fs: float = FS,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
) -> np.ndarray:
    """Full EEG preprocessing pipeline for a single sample.

    Pipeline: band-pass filter → FastICA artifact removal → z-score →
              sliding-window statistics (mean, std).

    Returns:
        ndarray of shape (time_frames, 2)
    """
    filtered = bandpass_filter(raw_vector, 0.5, 45.0, fs=fs)
    cleaned = run_fast_ica(filtered)
    normalized = zscore(cleaned)
    if normalized.ndim == 1:
        if normalized.size < window_size:
            padded = np.pad(normalized, (0, window_size - normalized.size), mode="edge")
            windowed = sliding_window_view(padded, window_shape=window_size)[::step]
        else:
            windowed = sliding_window_view(normalized, window_shape=window_size)[::step]
    else:
        windowed = sliding_window_view(normalized, window_shape=window_size, axis=0)[::step]
    return np.stack([windowed.mean(axis=1), windowed.std(axis=1)], axis=-1)


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
