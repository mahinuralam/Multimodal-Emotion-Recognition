# TMNet Multimodal Emotion Recognition

TMNet fuses synchronous speech and EEG streams to classify affective state. Speech waveforms undergo pre-emphasis, Wiener denoising, VAD trimming, and MFCC±Δ extraction (40 coefficients, hop 256). EEG vectors are band-pass filtered (0.5–45 Hz), cleaned with FastICA, z-scored, and converted to sliding-window statistics (113 frames × 2 features). Dynamic Time Warping aligns both modalities before resampling to a common temporal grid.

The model stack comprises:
- **Speech encoder:** CNN + BiLSTM stack that outputs 120-d frame embeddings.
- **EEG encoder:** stacked BiGRU layers operating on the 2-channel statistical sequence.
- **Fusion block:** Transformer cross-attention with LayerNorm and residual MLP for multimodal integration, followed by a 3-way softmax head (positive/neutral/negative).

## Usage
1. Open `TMNet_experiment copy.ipynb` and run the preprocessing cells to regenerate aligned tensors (`resampled_speech_sequences`, `resampled_eeg_sequences`, `resampled_labels`).
2. Execute the fusion training section to fine-tune encoders and the transformer head; monitor accuracy/curves stored in the notebook outputs.
3. Adapt downstream evaluation cells to export confusion matrices or saved weights (`.h5`) for deployment.
