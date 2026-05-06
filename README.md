# TMNet: Transformer-Fused Multimodal Emotion Recognition

[![Paper](https://img.shields.io/badge/Paper-ICT%20Express%202025-blue)](https://www.sciencedirect.com/science/article/pii/S2405959525000517)
[![License: Open Access](https://img.shields.io/badge/License-CC%20Open%20Access-green)](https://www.sciencedirect.com/science/article/pii/S2405959525000517)

TMNet is a deep learning framework that fuses **speech** and **EEG** brain signals to classify human emotional states into three categories — **Positive**, **Neutral**, and **Negative**. By integrating a CNN-BiLSTM speech encoder, a BiGRU EEG encoder, and a Transformer-based fusion block, TMNet achieves **98.89% accuracy**, outperforming all prior unimodal and multimodal baselines.

> **Published in:** ICT Express, Volume 11, Issue 4, August 2025, Pages 657–665
> **DOI:** [10.1016/j.icte.2025.04.007](https://doi.org/10.1016/j.icte.2025.04.007)

---

## Overview

<p align="center">
  <img src="assets/fig1_workflow.png" alt="Fig. 1 — Proposed multimodal emotion recognition workflow" width="800"/>
</p>
<p align="center"><em>Fig. 1. Proposed workflow of the multimodal emotion recognition framework.</em></p>

TMNet addresses the limitations of single-modal emotion recognition by combining complementary information from two physiological sources:

- **Speech stream** — irrelevant segments are removed using Voice Activity Detection (VAD) and background noise is reduced via spectral subtraction. Recordings are resampled to 22,050 Hz, and 40 MFCCs are extracted per frame (frame length 512, hop length 512). Feature matrices are standardised to a fixed length of 500 frames.
- **EEG stream** — raw signals are band-pass filtered to retain relevant frequencies, cleaned with Independent Component Analysis (ICA) to remove artifacts such as muscle movements and eye blinks, then normalised to zero mean and unit variance. Signals are segmented into overlapping 1-second windows (50% overlap) and Power Spectral Density (PSD) features are extracted in emotion-specific frequency bands: Neutral (5–25 Hz), Negative (up to 600 Hz), and Positive (above 600 Hz).

**Synchronisation via DTW:** A two-stage alignment approach first aligns timestamps by stimulus presentation times, then applies Dynamic Time Warping (DTW) to compensate for non-linear temporal variations. DTW was chosen over Connectionist Temporal Classification (CTC) — which suits single-modality variable-length sequences but is not designed for cross-modal synchronisation — and over temporal convolutions, which demand substantially more compute and complex training. The aligned joint feature vector is passed to a **Transformer acting as a meta-learner**, which applies multi-head self-attention to capture long-range inter-modal dependencies and generalises to unseen data through context-aware feature learning.

---

## Architecture

### EEG Artifact Removal

<p align="center">
  <img src="assets/fig2_artifact_removal.png" alt="Fig. 2 — EEG artifact removal" width="750"/>
</p>
<p align="center"><em>Fig. 2. Artifact removal from EEG signal of positive emotion: (a) original EEG signal with artifacts, (b) artifact-free signal after ICA processing.</em></p>

### TMNet Model Architecture

<p align="center">
  <img src="assets/fig3_architecture.png" alt="Fig. 3 — TMNet multimodal architecture" width="800"/>
</p>
<p align="center"><em>Fig. 3. Proposed TMNet multimodal architecture highlights the convergence of the Speech and EEG signals into the Transformer for complex model feature extraction and prediction.</em></p>

The model stack consists of three independently trained then jointly fused components:

| Component | Architecture | Output |
|-----------|-------------|--------|
| **Speech Encoder (CNN-BiLSTM)** | 4× Conv2D + BatchNorm + MaxPool → TimeDistributed Flatten → Dropout → Dense(512, ReLU) → Bidirectional LSTM(128) + BatchNorm + Dropout → Dense(5, softmax) | 5-class speech emotion logits |
| **EEG Encoder (BiGRU)** | 3× Bidirectional GRU (64→128→64) + BatchNorm + Dropout → Flatten → Dense(128, ReLU) + Dropout → Dense(3, softmax) | 3-class EEG emotion logits |
| **Fusion Block (Transformer)** | Concatenate outputs → Transformer encoder (multi-head self-attention, Q/K/V projections, FFN with ReLU, LayerNorm + residual, Dropout) → Dense + Dropout → Dense(3, softmax) | 3-class emotion prediction |

**Speech model input:** 3D tensor of shape `(max_len=500, num_mfcc=40, 1)`.
**EEG model input:** 3D tensor of shape `(sequence_length, eeg_features)`.
**Fusion:** Logit outputs from both frozen encoders are concatenated into a joint feature vector `C`, then fed into the Transformer encoder. Multi-head self-attention computes attention scores as `softmax(QKᵀ / √d_k) · V`, where `d_k` is the key dimension for gradient stability. A feed-forward network with ReLU activation follows, with layer normalisation and dropout applied throughout.

---

## Results

### Speech Model: CNN-BiLSTM Evaluation

The CNN-BiLSTM model is evaluated against state-of-the-art speech emotion recognition approaches on the same multi-corpus benchmark (RAVDESS + SAVEE + TESS + CREMA-D):

| Model | Datasets | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|-------|----------|:---:|:---:|:---:|:---:|
| CNN-MLP [20] | RAVDESS | 86.40 | 86.50 | 86.30 | 86.32 |
| CNN [21] | RAVDESS | 71.61 | 71.80 | 71.50 | 71.65 |
| CNN [22] | R+S+T+C | 92.60 | 92.70 | 92.50 | 92.60 |
| CNN [23] | R+S+T+C | 90.00 | 90.10 | 89.90 | 89.91 |
| **CNN-BiLSTM (ours)** | **R+S+T+C** | **95.33** | **95.40** | **95.30** | **95.35** |

*R: RAVDESS, S: SAVEE, T: TESS, C: CREMA-D*

### EEG Model: BiGRU Evaluation

<p align="center">
  <img src="assets/fig5_confusion_matrix.png" alt="Fig. 5 — Confusion matrix comparison of BiGRU vs state-of-the-art" width="850"/>
</p>
<p align="center"><em>Fig. 5. Comparison of the proposed BiGRU with state-of-the-art models (BiGRU, InceptionV3, ResNet50, VGG16).</em></p>

The confusion matrices show that BiGRU effectively distinguishes Neutral (150/153), Positive (135/140), and Negative (132/133) emotions with minimal misclassification.

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|-------|:---:|:---:|:---:|:---:|
| DeepMaxout [24] | 90.84 | 91.00 | 90.50 | 90.75 |
| SVM [10] | 90.00 | 90.20 | 89.80 | 89.88 |
| GRU [25] | 96.95 | 97.00 | 96.90 | 96.51 |
| **BiGRU (ours)** | **97.25** | **97.40** | **97.20** | **97.21** |

### Multimodal Fusion: TMNet Evaluation

| Model | Method | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|-------|--------|:---:|:---:|:---:|:---:|
| Multimodal Transformer [5] | Transformer, SVM Classifier | 91.79 | 92.00 | 91.50 | 91.70 |
| Late Fusion [26] | CNN, GRU, Encoder | 95.39 | 95.50 | 95.30 | 95.38 |
| Feature Level Fusion [27] | SVM | 87.44 | 87.60 | 87.30 | 87.45 |
| Decision Level Fusion [4] | kNN, SVM | 96.24 | 96.12 | 95.91 | 95.87 |
| Hierarchical LSTM Fusion with Self-Attention [28] | LSTM, Attention | 94.86 | 93.96 | 94.89 | 94.88 |
| Multi CNN Fusion [29] | DeepVANet, CNN | 95.61 | 95.11 | 95.08 | 95.01 |
| **TMNet (ours)** | **CNN-BiLSTM, BiGRU, Transformer** | **98.89** | **99.00** | **98.80** | **98.88** |

### Noise and Artifact Robustness

TMNet was stress-tested under progressively degraded EEG input conditions:

| Condition | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|-----------|:---:|:---:|:---:|:---:|
| Baseline (clean) | 98.89 | 99.00 | 98.80 | 98.88 |
| Low Noise & Artifacts | 95.20 | — | — | — |
| Medium Noise & Artifacts | 90.75 | — | 90.30 | — |
| High Noise & Artifacts | 85.30 | 86.10 | — | 85.40 |

### Ablation Study

Each preprocessing component was individually removed to quantify its contribution:

| Configuration | Accuracy (%) |
|--------------|:---:|
| Full TMNet (baseline) | 98.89 |
| Without ICA | 95.12 |
| Without Bandpass Filter | 96.08 |
| Without MFCCs | 94.01 |
| DTW replaced with Simple Timestamp Alignment | 93.12 |

### Architecture Efficiency Comparison

| Speech Model | EEG Model | MAE | Parameters | Inference Time (s) |
|-------------|-----------|:---:|:---:|:---:|
| **BiGRU** | **CNN-BiLSTM** | **0.05** | **5.6M** | **0.12** |
| GRU | LSTM | 0.12 | 4.8M | 0.18 |
| CNN | CNN-LSTM | 0.15 | 3.9M | 0.15 |
| RNN | MLP | 0.35 | 1.4M | 0.08 |
| BiGRU | DCNN | 0.18 | 5.5M | 0.20 |
| CNN-GRU | CNN-BiLSTM | 0.10 | 7.3M | 0.25 |
| LSTM | CNN-BiLSTM | 0.13 | 6.4M | 0.22 |
| DCNN-BiGRU | DCNN-BiLSTM | 0.90 | 9.5M | 0.30 |

The BiGRU + CNN-BiLSTM combination achieves the lowest MAE (0.05) with a favourable balance of 5.6M parameters and 0.12s inference time.

---

## Repository Structure

```
Multimodal-Emotion-Recognition/
├── src/
│   ├── preprocessing/
│   │   ├── speech.py          # VAD, spectral subtraction, MFCC extraction
│   │   └── eeg.py             # Band-pass filter, FastICA, normalisation, PSD features
│   ├── models/
│   │   ├── speech_model.py    # CNN + Bidirectional LSTM encoder
│   │   ├── eeg_model.py       # Stacked Bidirectional GRU encoder
│   │   └── fusion_model.py    # TMNet Transformer-based fusion model
│   ├── utils/
│   │   ├── augmentation.py    # SpecAugment-style time/frequency masking + Gaussian noise
│   │   └── alignment.py       # DTW alignment and temporal resampling
│   └── data/
│       └── loader.py          # Dataset loading utilities
├── assets/                    # Paper figures (see assets/README.md)
├── Model/                     # Saved trained model weights
│   ├── my_model.h5            # Speech CNN-BiLSTM
│   ├── my_model_gru.h5        # EEG BiGRU
│   └── multimodal_model.keras # TMNet fusion model
├── Output/                    # Sample prediction visualisations
├── TMNet.ipynb                # End-to-end experiment notebook
└── requirements.txt
```

---

## Datasets

### Speech Datasets

Seven thousand six hundred sixty-four audio recordings are sourced from four publicly available, ethically approved corpora:

| Dataset | Emotions covered | Format |
|---------|-----------------|--------|
| [RAVDESS](https://zenodo.org/record/1188976) | Calm, Happy, Neutral, Sad, Angry | `.wav` |
| [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/) | Happy, Neutral, Sad, Angry | `.wav` |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) | Happy, Neutral, Sad, Angry | `.wav` |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Happy, Neutral, Sad, Angry | `.wav` |

**Sample counts per emotion used in the paper:**

| Emotion | CREMA-D | SAVEE | RAVDESS | TESS |
|---------|:---:|:---:|:---:|:---:|
| Calmness | 0 | 0 | 192 | 0 |
| Happiness | 1271 | 60 | 192 | 400 |
| Neutral | 1087 | 120 | 96 | 400 |
| Sadness | 1271 | 60 | 192 | 400 |
| Anger | 1271 | 60 | 192 | 400 |

Organise downloaded files under `Datasets/Speech/<Emotion>/` (sub-directories: `Angry`, `Calm`, `Neutral`, `Happy`, `Sad`).

### EEG Dataset

EEG signals were recorded from two participants (one male, one female) using dry electrodes and a **Muse EEG headband**, capturing three emotional states: Positive, Neutral, and Negative. The publicly available dataset ([Bird et al., 2019](https://doi.org/10.1155/2019/4316548)) is stored as a single CSV file with 750 FFT-band features per sample.

Place it at: `Datasets/EEG/emotions.csv` (columns: `fft_0_b` … `fft_749_b`, `label`).

### Emotion Mapping (Speech → 3-class)

| Speech Emotion | General Class |
|----------------|--------------|
| Angry | Negative |
| Sad | Negative |
| Happy | Positive |
| Calm | Positive |
| Neutral | Neutral |

The data was split **70% training / 15% validation / 15% testing** with stratified sampling.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mahinuralam/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare datasets

Download the speech and EEG datasets (links above) and organise them following the structure in the [Datasets](#datasets) section. Then update the path variables at the top of `TMNet.ipynb` to point to your local copies.

---

## Usage

### Running the full experiment

Open and execute the notebook from the repository root:

```bash
jupyter notebook TMNet.ipynb
```

Run the cells in order:

| Step | Description |
|------|-------------|
| 1. Speech Preprocessing | Load audio from RAVDESS/SAVEE/TESS/CREMA-D, extract 40 MFCCs, pad to 500 frames, apply SpecAugment augmentation |
| 2. EEG Preprocessing | Load `emotions.csv`, band-pass filter, FastICA artifact removal, build sliding-window feature sequences |
| 3. Speech Model Training | Train CNN-BiLSTM (5-class), save to `Model/my_model.h5` |
| 4. EEG Model Training | Train BiGRU (3-class), save to `Model/my_model_gru.h5` |
| 5. Multimodal Alignment | DTW-align speech and EEG sequences, resample to common temporal length |
| 6. Fusion Training | Freeze encoders, train TMNet Transformer fusion (3-class), save to `Model/multimodal_model.keras` |
| 7. Evaluation | Generate confusion matrices, classification reports, and TensorBoard curves |

### Using the Python modules directly

```python
from src.preprocessing.speech import preprocess_speech_file
from src.preprocessing.eeg import preprocess_eeg_sample, Transform_data
from src.models.speech_model import build_speech_model
from src.models.eeg_model import build_eeg_model
from src.models.fusion_model import build_fusion_model
from src.utils.alignment import dtw_align_pair, resample_aligned_sequences
from src.utils.augmentation import augment_dataset
from src.data.loader import load_speech_dataset, load_eeg_dataset

# Load datasets
X_speech, y_speech, file_paths = load_speech_dataset("Datasets/Speech/")
eeg_df = load_eeg_dataset("Datasets/EEG/emotions.csv")
X_eeg, Y_eeg, X_eeg_raw = Transform_data(eeg_df)

# Build and train individual encoders
speech_model = build_speech_model(max_len=500, num_mfcc=40, num_classes=5)
eeg_model = build_eeg_model(input_timesteps=X_eeg.shape[1], input_features=X_eeg.shape[2])

# Build the TMNet multimodal fusion model (freezes both encoders)
fusion_model = build_fusion_model(speech_model, eeg_model, num_classes=3)
```

### Monitoring training with TensorBoard

```bash
tensorboard --logdir logs_multimodal
```

Available log directories:

| Directory | Contents |
|-----------|----------|
| `logs/` | Initial speech model run |
| `logs_2/` | EEG BiGRU training |
| `logs_improved/` | CNN-BiLSTM with full preprocessing |
| `logs_multimodal/` | TMNet fusion training |

---

## Citation

If you use this code or the TMNet architecture in your research, please cite:

```bibtex
@article{alam2025tmnet,
  title     = {TMNet: Transformer-fused multimodal framework for emotion recognition via EEG and speech},
  author    = {Alam, Md Mahinur and Dini, Mohamed A and Kim, Dong-Seong and Jun, Taesoo},
  journal   = {ICT Express},
  volume    = {11},
  number    = {4},
  pages     = {657--665},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.icte.2025.04.007},
  url       = {https://www.sciencedirect.com/science/article/pii/S2405959525000517}
}
```

---

## Acknowledgements

This research was supported by:
- Innovative Human Resource Development for Local Intellectualization Program, South Korea (IITP-2025-RS-2020-II201612) — 34%
- Basic Science Research Program, South Korea (2018R1A6A1A03024003) through NRF — 33%
- Information Technology Research Center (ITRC) Program, South Korea (IITP-2025-RS-2024-00438430) — 33%

---

## License

This project is published as open access under a Creative Commons license. Please refer to the [published paper](https://www.sciencedirect.com/science/article/pii/S2405959525000517) for full details on methodology and experimental setup.
