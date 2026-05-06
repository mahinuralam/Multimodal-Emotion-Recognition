import numpy as np
import librosa

TARGET_SR = 22050   # as stated in paper
NUM_MFCC = 40
MAX_LEN = 500
HOP_LENGTH = 512    # as stated in paper (hop length 512)
N_FFT = 512         # as stated in paper (frame length 512)


def spectral_subtraction(
    y: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    noise_frames: int = 10,
    alpha: float = 2.0,
    beta: float = 0.01,
) -> np.ndarray:
    """Reduce background noise via spectral subtraction (as described in paper).

    Estimates noise from the first `noise_frames` STFT frames (assumed to be
    silence/noise-only) and subtracts it from the full magnitude spectrum.

    Args:
        y:            Input audio signal.
        n_fft:        FFT window size.
        hop_length:   Hop length between frames.
        noise_frames: Number of leading frames used to estimate the noise floor.
        alpha:        Over-subtraction factor (higher = more aggressive removal).
        beta:         Spectral floor to prevent negative values.

    Returns:
        Denoised audio signal reconstructed via iSTFT.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    n_frames = min(noise_frames, magnitude.shape[1])
    noise_est = np.mean(magnitude[:, :n_frames], axis=1, keepdims=True)

    magnitude_clean = np.maximum(magnitude - alpha * noise_est, beta * magnitude)
    stft_clean = magnitude_clean * np.exp(1j * phase)
    return librosa.istft(stft_clean, hop_length=hop_length)


def apply_vad(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    """Trim leading/trailing silence using energy-based VAD."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size else y


def preprocess_speech_file(
    file_path: str,
    target_sr: int = TARGET_SR,
    n_mfcc: int = NUM_MFCC,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Load and preprocess a single audio file following the paper pipeline.

    Pipeline (Section 3, paper):
        1. Load and resample to 22,050 Hz
        2. VAD — remove irrelevant silence segments
        3. Spectral subtraction — reduce background noise
        4. MFCC extraction — 40 coefficients, frame/hop length 512

    Returns:
        ndarray of shape (timesteps, n_mfcc)
    """
    y, sr = librosa.load(file_path, sr=target_sr)
    y = apply_vad(y)
    y = spectral_subtraction(y, n_fft=n_fft, hop_length=hop_length)
    if y.size == 0:
        y = np.zeros(int(target_sr * 0.5))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length,
                                  n_fft=n_fft)
    return mfcc.T  # (timesteps, n_mfcc)


def extract_features(
    file_path: str,
    sr: int = TARGET_SR,
    n_mfcc: int = NUM_MFCC,
) -> np.ndarray:
    """Basic MFCC extraction used during initial dataset loading."""
    y, sr = librosa.load(file_path, sr=sr)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def load_and_pad_mfcc(
    file_path: str,
    max_len: int = MAX_LEN,
    n_mfcc: int = NUM_MFCC,
) -> np.ndarray | None:
    """Extract MFCCs and pad/truncate to max_len frames.

    Returns ndarray of shape (max_len, n_mfcc), or None if file exceeds max_len.
    """
    mfccs = extract_features(file_path, n_mfcc=n_mfcc)
    if mfccs.shape[1] > max_len:
        return None
    pad_width = max_len - mfccs.shape[1]
    return np.pad(mfccs.T, pad_width=((0, pad_width), (0, 0)), mode="constant")
