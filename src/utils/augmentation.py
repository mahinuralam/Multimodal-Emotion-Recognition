import random
import numpy as np


def apply_time_mask(mfcc: np.ndarray, max_mask_size: int = 40) -> np.ndarray:
    """Zero out a random contiguous block of time frames (SpecAugment-style)."""
    mfcc = mfcc.copy()
    mask_size = np.random.randint(1, max_mask_size)
    start = np.random.randint(0, mfcc.shape[0] - mask_size)
    mfcc[start : start + mask_size, :] = 0
    return mfcc


def apply_frequency_mask(mfcc: np.ndarray, max_mask_size: int = 8) -> np.ndarray:
    """Zero out a random contiguous block of MFCC frequency bins."""
    mfcc = mfcc.copy()
    mask_size = np.random.randint(1, max_mask_size)
    start = np.random.randint(0, mfcc.shape[1] - mask_size)
    mfcc[:, start : start + mask_size] = 0
    return mfcc


def add_gaussian_noise(mfcc: np.ndarray, noise_factor: float = 0.015) -> np.ndarray:
    """Add zero-mean Gaussian noise to MFCC features."""
    noise = np.random.normal(0, noise_factor, mfcc.shape)
    return mfcc + noise


def augment_mfcc(sample: np.ndarray) -> np.ndarray:
    """Stochastically apply time mask, frequency mask, and Gaussian noise."""
    augmented = sample.copy()
    if random.random() < 0.6:
        augmented = add_gaussian_noise(augmented)
    if random.random() < 0.5:
        augmented = apply_time_mask(augmented)
    if random.random() < 0.5:
        augmented = apply_frequency_mask(augmented)
    return augmented


def augment_dataset(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Double the training set by creating one augmented copy per sample.

    Returns:
        Shuffled concatenation of original and augmented samples.
    """
    augmented_samples = np.array(
        [augment_mfcc(sample) for sample in X_train], dtype=np.float32
    )
    X_aug = np.concatenate([X_train.astype(np.float32), augmented_samples], axis=0)
    y_aug = np.concatenate([y_train, y_train], axis=0)

    indices = np.arange(X_aug.shape[0])
    np.random.shuffle(indices)
    return X_aug[indices], y_aug[indices]
