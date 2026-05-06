import numpy as np
import librosa
from scipy import signal


def first_stage_align(
    speech_paths: np.ndarray,
    eeg_signals: np.ndarray,
    speech_labels: np.ndarray,
    eeg_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-stage alignment: pair speech and EEG samples by shared emotion label.

    The paper describes a two-stage synchronisation approach:
        Stage 1 — align by stimulus presentation timestamps (or, when explicit
                  timestamps are unavailable, by matching emotion labels so that
                  each speech sample is paired with an EEG sample of the same
                  emotional category).
        Stage 2 — DTW refinement of the paired sequences (see dtw_align_pair).

    Args:
        speech_paths:  Array of speech file paths, shape (N_speech,).
        eeg_signals:   Array of EEG raw feature vectors, shape (N_eeg, features).
        speech_labels: Integer emotion labels for speech samples, shape (N_speech,).
        eeg_labels:    Integer emotion labels for EEG samples, shape (N_eeg,).

    Returns:
        aligned_speech_paths: shape (M,)
        aligned_eeg_signals:  shape (M, features)
        aligned_labels:       shape (M,)  — shared emotion label per pair
    """
    aligned_speech, aligned_eeg, aligned_labels = [], [], []

    # Group indices by emotion class
    speech_by_class: dict[int, list[int]] = {}
    for idx, lbl in enumerate(speech_labels):
        speech_by_class.setdefault(int(lbl), []).append(idx)

    eeg_by_class: dict[int, list[int]] = {}
    for idx, lbl in enumerate(eeg_labels):
        eeg_by_class.setdefault(int(lbl), []).append(idx)

    for cls in sorted(set(speech_by_class) & set(eeg_by_class)):
        s_indices = speech_by_class[cls]
        e_indices = eeg_by_class[cls]
        n_pairs = min(len(s_indices), len(e_indices))
        for i in range(n_pairs):
            aligned_speech.append(speech_paths[s_indices[i]])
            aligned_eeg.append(eeg_signals[e_indices[i]])
            aligned_labels.append(cls)

    return (
        np.array(aligned_speech),
        np.array(aligned_eeg, dtype=np.float32),
        np.array(aligned_labels),
    )


def dtw_align_pair(
    speech_seq: np.ndarray, eeg_seq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Second-stage alignment: refine a matched pair using Dynamic Time Warping.

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
