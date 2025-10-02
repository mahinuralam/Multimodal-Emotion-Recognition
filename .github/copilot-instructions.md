# GitHub Copilot Instructions for Multimodal Emotion Recognition

## Project shape
- Core work happens in `TMNet.ipynb`; treat it as the source of truth for preprocessing, modeling, and evaluation.
- Saved artifacts live in `Model/` (`my_model.h5`, `my_model_gru.h5`, `multimodal_model.keras`) and TensorBoard logs in `logs*` directories.
- Datasets are expected under `Datasets/Speech/<Emotion>` (Angry/Calm/Neutral/Happy/Sad) and `Datasets/EEG/emotions.csv`; code calls `os.path.expanduser` on absolute paths.

## Speech pipeline
- Reuse `extract_features`, `preprocess_speech_file`, and the SpecAugment-style helpers (`apply_time_mask`, `apply_frequency_mask`, `add_gaussian_noise`) when touching audio data.
- Speech samples are padded/truncated to `max_len=500` frames with `num_mfcc=40`; maintain this shape when adding models so downstream layers (CNN → TimeDistributed → BiLSTM) stay compatible.
- `augment_mfcc` doubles the training set; any new augmentation must preserve `(500, 40)` before the channel expansion step.

## EEG pipeline
- `Transform_data` standardizes EEG FFT features then one-hot encodes labels; keep label order `NEUTRAL=0, POSITIVE=1, NEGATIVE=2` in sync with `speech_general_mapping`.
- Sequence shaping occurs via `preprocess_eeg_sample` (band-pass → FastICA → sliding windows). Maintain `window_size=128` and `step=64` so GRU inputs stay `(time, 2)`.
- The Bidirectional GRU stack (`create_improved_model`) feeds a 3-way softmax; when modifying, keep dropout + BatchNorm cadence to avoid overfitting noted in prior experiments.

## Multimodal fusion
- Speech and EEG branches are first trained separately, then frozen via `clone_with_prefix`; any new branch must be cloned before fusion to avoid weight sharing bugs.
- Alignment relies on `dtw_align_pair` + `resample_aligned_sequences`; keep DTW outputs in `dtype=np.float32` and resample to `target_sequence_length` derived from the 75th percentile heuristic.
- Fusion training slices each modality to a common `min_train`/`min_test` count; adjust both if you introduce new sampling strategies.

## Evaluation & logging
- Use existing callbacks: TensorBoard log dirs (`./logs_improved`, `./logs_2`, `./logs_multimodal`) and EarlyStopping tuned on `val_accuracy`.
- Confusion matrices are generated post-training with seaborn heatmaps; follow this pattern for any new experiments to keep outputs consistent.
- Persist new weights alongside existing models in `Model/` and note filenames inside this document when added.

## Working tips
- Run notebooks from the repo root so relative paths to `Datasets/` and `logs*` resolve.
- When integrating new modalities or labels, update `speech_general_mapping` and regenerate `resampled_*` tensors to stay synchronized.
- Keep saved tensors (`resampled_speech_sequences`, `resampled_eeg_sequences`, `fusion_labels_onehot`) immutable between runs unless the preprocessing steps change; many cells assume they already exist to avoid recomputation.
- Prefer extending helper functions instead of inlining new preprocessing so repeated notebook runs remain idempotent.
