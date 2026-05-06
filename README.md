# TMNet: Transformer-Fused Multimodal Emotion Recognition

[![Paper](https://img.shields.io/badge/Paper-ICT%20Express%202025-blue)](https://www.sciencedirect.com/science/article/pii/S2405959525000517)
[![License: Open Access](https://img.shields.io/badge/License-CC%20Open%20Access-green)](https://www.sciencedirect.com/science/article/pii/S2405959525000517)

TMNet fuses **speech** and **EEG** signals to classify emotions into **Positive**, **Neutral**, and **Negative** using a CNN-BiLSTM speech encoder, BiGRU EEG encoder, and Transformer-based fusion — achieving **98.89% accuracy**.

> **ICT Express**, Vol. 11, No. 4, pp. 657–665, 2025 · [10.1016/j.icte.2025.04.007](https://doi.org/10.1016/j.icte.2025.04.007)

---

## Overview

<p align="center">
  <img src="assets/fig1_workflow.png" alt="Fig. 1 — Proposed multimodal emotion recognition workflow" width="800"/>
</p>
<p align="center"><em>Fig. 1. Proposed workflow of the multimodal emotion recognition framework.</em></p>

- **Speech** — VAD denoising, spectral subtraction, resampled to 22,050 Hz, 40 MFCCs extracted (hop 512), padded to 500 frames.
- **EEG** — band-pass filtered, ICA artifact removal, z-score normalised, segmented into 1-second overlapping windows, PSD features extracted per emotion-specific frequency band.
- **Fusion** — Dynamic Time Warping (DTW) aligns both modalities temporally; a Transformer meta-learner captures inter-modal dependencies via multi-head self-attention.

---

## Architecture

<p align="center">
  <img src="assets/fig3_architecture.png" alt="Fig. 3 — TMNet architecture" width="800"/>
</p>
<p align="center"><em>Fig. 3. TMNet multimodal architecture: Speech CNN-BiLSTM and EEG BiGRU encoders fused via Transformer.</em></p>

| Component | Architecture | Output |
|-----------|-------------|--------|
| **Speech Encoder** | 4× Conv2D + BatchNorm + MaxPool → TimeDistributed Flatten → Dense(512) → BiLSTM(128) | 5-class logits |
| **EEG Encoder** | 3× BiGRU (64→128→64) + BatchNorm + Dropout → Dense(128) | 3-class logits |
| **Fusion (Transformer)** | Concat → Multi-Head Self-Attention + FFN + LayerNorm → Dense + Dropout → Softmax | 3-class prediction |

---

## Results

<p align="center">
  <img src="assets/fig5_confusion_matrix.png" alt="Fig. 5 — Confusion matrix comparison" width="850"/>
</p>
<p align="center"><em>Fig. 5. Comparison of BiGRU with state-of-the-art models.</em></p>

| Model | Method | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|-------|--------|:---:|:---:|:---:|:---:|
| Multimodal Transformer [5] | Transformer, SVM | 91.79 | 92.00 | 91.50 | 91.70 |
| Late Fusion [26] | CNN, GRU, Encoder | 95.39 | 95.50 | 95.30 | 95.38 |
| Feature Level Fusion [27] | SVM | 87.44 | 87.60 | 87.30 | 87.45 |
| Decision Level Fusion [4] | kNN, SVM | 96.24 | 96.12 | 95.91 | 95.87 |
| Hierarchical LSTM + Self-Attention [28] | LSTM, Attention | 94.86 | 93.96 | 94.89 | 94.88 |
| Multi CNN Fusion [29] | DeepVANet, CNN | 95.61 | 95.11 | 95.08 | 95.01 |
| **TMNet (ours)** | **CNN-BiLSTM, BiGRU, Transformer** | **98.89** | **99.00** | **98.80** | **98.88** |

---

## Repository Structure

```
Multimodal-Emotion-Recognition/
├── src/
│   ├── preprocessing/     # speech.py, eeg.py
│   ├── models/            # speech_model.py, eeg_model.py, fusion_model.py
│   ├── utils/             # augmentation.py, alignment.py
│   └── data/              # loader.py
├── assets/                # Paper figures
├── Model/                 # Saved weights (.h5, .keras)
├── TMNet.ipynb            # End-to-end experiment notebook
└── requirements.txt
```

---

## Datasets

- **Speech:** RAVDESS, SAVEE, TESS, CREMA-D (7,664 recordings). Place under `Datasets/Speech/<Emotion>/`.
- **EEG:** [Bird et al. 2019](https://doi.org/10.1155/2019/4316548) — Muse headband recordings. Place CSV at `Datasets/EEG/emotions.csv`.

---

## Installation & Usage

```bash
git clone https://github.com/mahinuralam/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
pip install -r requirements.txt
jupyter notebook TMNet.ipynb
```

Run notebook cells in order: speech preprocessing → EEG preprocessing → train CNN-BiLSTM → train BiGRU → DTW alignment → train TMNet fusion → evaluation.

---

## Citation

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
