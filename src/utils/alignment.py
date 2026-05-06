import numpy as np
import librosa
from scipy import signal


def dtw_align_pair(
    speech_seq: np.ndarray, eeg_seq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align a speech and EEG feature sequence pair using Dynamic Time Warping.

    Both sequences are projected to 1-D norm vectors before DTW so that the
    alignment path is computed on comparable scalar representations.

    Returns:
        aligned_speech: speech frames reindexed along the DTW warping path
        aligned_eeg:    EEG frames reindexed along the DTW warping path
    """
    speech_arr = np.asarray(speech_seq, dtype=np.float32)
    eeg_arr = np.asarray(eeg_seq, dtype=np.float32)

    if speech_arr.ndim == 1:
        speech_arr = speech_arr[:, None]
    elif speech_arr.ndim > 2:
        speech_arr = speech_arr.reshape(speech_arr.shape[0], -1)

    if eeg_arr.ndim == 1:
        eeg_arr = eeg_arr[:, None]
    elif eeg_arr.ndim > 2:
        eeg_arr = eeg_arr.reshape(eeg_arr.shape[0], -1)

    speech_norm = np.linalg.norm(speech_arr, axis=1, keepdims=True).T
    eeg_norm = np.linalg.norm(eeg_arr, axis=1, keepdims=True).T

    _, wp = librosa.sequence.dtw(X=speech_norm, Y=eeg_norm, metric="euclidean")
    wp = np.array(wp[::-1])

    return speech_arr[wp[:, 0]], eeg_arr[wp[:, 1]]


def resample_aligned_sequences(
    speech_seq: np.ndarray, eeg_seq: np.ndarray, target_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample DTW-aligned sequences to a common temporal length."""
    speech_r = signal.resample(speech_seq.astype(np.float32), target_len, axis=0)
    eeg_r = signal.resample(eeg_seq.astype(np.float32), target_len, axis=0)
    return speech_r.astype(np.float32), eeg_r.astype(np.float32)


def compute_target_length(dtw_lengths: np.ndarray, percentile: int = 75) -> int:
    """Derive the resampling target from the distribution of DTW output lengths."""
    return int(np.clip(np.percentile(dtw_lengths, percentile), 80, 320))
