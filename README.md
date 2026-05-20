# ECG Arrhythmia Classification

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

1D CNN for ECG arrhythmia classification with SMOTE and focal loss, validated on a Samsung Galaxy Watch 8 Classic via a novel PDF digitization pipeline.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Test Accuracy | 98.44% |
| Macro F1-Score | 0.9153 |
| Weighted F1-Score | 0.9843 |
| AUC-ROC | 0.9925 |
| Model Parameters | 307,333 |
| Training Time | ~8 minutes on Google Colab T4 GPU |
| Baseline (Random Forest) | 29.27% |

---

## Annotated ECG — Galaxy Watch 8 Classification

![ECG Classification](figures/ecg_classification.png)

**45 beats detected at 93.8 bpm — Diagnosis: Normal Sinus Rhythm (97.8% Normal)**

---

## The Problem This Project Solves

The MIT-BIH Arrhythmia Database has a severe class imbalance. Normal beats make up 82.8% of all training samples. The Fusion class has only 641 samples — a 113:1 ratio.

A model that always predicts Normal gets 82.8% accuracy while never detecting a single arrhythmia. Standard training with class weights achieves 29.27% — worse than the naive baseline — because the model learns to guess minority classes aggressively without actually learning what they look like.

This project solves it with two techniques working together:

**SMOTE** generates synthetic minority class samples by interpolating between real examples in the 187-dimensional feature space. Unlike simple duplication, SMOTE explores the actual shape of each minority class boundary.

**Focal Loss** (γ=2.0, α=0.25) down-weights the gradient contribution of easy, well-classified Normal beats. A Normal beat at 95% confidence contributes only 0.25% of its standard cross-entropy loss. A hard Fusion beat at 20% confidence retains 64%.

Neither works alone. SMOTE without focal loss still has gradient dominated by 72,471 Normal beats. Focal loss without SMOTE has almost no genuine minority class data to learn from. Together they are synergistic.

---

## Per-Class Performance

| Class | Description | Precision | Recall | F1 | Support |
|-------|-------------|-----------|--------|----|---------|
| N | Normal sinus beat | 0.99 | 0.99 | 0.99 | 18,118 |
| S | Supraventricular ectopic | 0.88 | 0.79 | 0.83 | 556 |
| V | Ventricular ectopic | 0.98 | 0.95 | 0.96 | 1,448 |
| F | Fusion beat | 0.72 | 0.90 | 0.80 | 162 |
| Q | Unknown/Paced | 0.98 | 0.99 | 0.99 | 1,608 |

Fusion class recall improved from 0.00 (baseline) to 0.90. Q class from 0.00 to 0.99.

---

## Confusion Matrix and Training Curves

![Training Curves](figures/fig_training_curves.png)

![Confusion Matrix](figures/fig_confusion_matrix.png)

---

## Model Architecture

```
Input (187 × 1)
      ↓
Conv1D Block 1 — 64 filters, kernel 5
      ↓
Conv1D Block 2 — 128 filters, kernel 5
      ↓
Conv1D Block 3 — 256 filters, kernel 3
      ↓
Conv1D Block 4 — 128 filters, kernel 3
      ↓
Global Average Pooling
      ↓
Dense(256) → Dropout(0.4)
      ↓
Dense(128) → Dropout(0.3)
      ↓
Dense(5) → Softmax
```

Each convolutional block: Conv1D → Batch Normalisation → ReLU → MaxPooling → Dropout(0.2)

**307,333 parameters. Trained from scratch. No pre-training or transfer learning.**

Kernel size 5 in early blocks covers 40ms at 125 Hz — enough to span the complete P wave, QRS complex, and T wave in a single pass.

Global Average Pooling replaces Flatten, reducing parameters and providing implicit regularisation over the time dimension.

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimiser | Adam, initial lr 0.001 |
| Loss | Focal Loss (γ=2.0, α=0.25) |
| Batch size | 128 |
| Epochs | 46 of 80 (early stopping) |
| LR scheduler | ReduceLROnPlateau — halved 6 times |
| Final LR | 0.0000078 |
| Hardware | Google Colab T4 GPU |

---
## Focal Loss

![Focal Loss](figures/fig_focal_loss.png)

Focal loss down-weights easy, well-classified examples so the model focuses on hard 
minority class examples. With γ=2.0, a Normal beat at 95% confidence contributes only 
0.25% of standard cross-entropy loss. A Fusion beat at 20% confidence retains 64%.

---

## SMOTE Augmentation

| Class | Before SMOTE | After SMOTE | Change |
|-------|-------------|-------------|--------|
| N (Normal) | 72,471 | 72,471 | unchanged |
| S (SVEB) | 2,223 | 5,000 | +125% |
| V (VEB) | 5,788 | 6,000 | +4% |
| F (Fusion) | 641 | 3,000 | +368% |
| Q (Unknown) | 6,431 | 7,000 | +9% |
| **Total** | **87,554** | **93,471** | balanced |

*Training set after SMOTE and validation split: 84,123 samples.*

---

## Real-World Validation: Samsung Galaxy Watch 8

Most ECG papers stop at benchmark evaluation. This project went further.

### The Data Access Problem

I tried to access the raw ECG signal from a Samsung Galaxy Watch 8 Classic via ADB on Android 13. A systematic investigation confirmed:

- ADB wireless debugging was enabled and device was successfully paired
- Full file system search was performed
- Result: ECG data in protected sandbox — inaccessible without rooting

Samsung Health Monitor only exports a formatted PDF report. No raw signal, no CSV, no API. This is a documented barrier for consumer wearable ECG research.

### The PDF Digitization Pipeline

A five-stage computer vision pipeline was built to reconstruct the ECG signal from the PDF export:

1. **Render PDF at 300 DPI** — 11.8 pixels per mm, enough to track the 1-2px trace
2. **HSV colour masking** — isolates Samsung Health's orange ECG trace (hue 5° to 25°)
3. **Strip detection** — finds three 10-second ECG strips by row density analysis
4. **Pixel centroid tracking** — column-wise centroid extraction with linear interpolation
5. **FFT resampling** — each strip resampled to 1,250 samples at 125 Hz

Total: 3,750 samples representing 30 seconds of ECG at 125 Hz.

### The Window Overlap Problem

At high heart rates (>80 bpm), the RR interval is shorter than the standard MIT-BIH half-window of 93 samples. A standard 187-sample window captures two beats instead of one, causing the model to misclassify everything as VEB.

**Solution:** Extract 80% of the actual RR interval around each peak and resample to 187 samples. This preserves the QRS morphology within a single beat window without overlapping adjacent beats.

### Galaxy Watch Results

![Beat Windows](figures/ecg_beat_windows.png)

| Class | Beats | Percentage | Mean Confidence |
|-------|-------|------------|-----------------|
| N (Normal) | 44 | 97.8% | 84.3% |
| V (VEB) | 1 | 2.2% | 85.7% |
| **Total** | **45** | — | **84.3%** |

**Detected BPM: 93.8 — Diagnosis: Normal Sinus Rhythm**

The single VEB occurs at a strip boundary where PDF rendering attenuates the signal. Mean confidence of 84.3% vs near-perfect on MIT-BIH directly quantifies the domain gap introduced by PDF reconstruction.

Result is clinically consistent with Samsung Health's own classification.

---

## Dataset

MIT-BIH Arrhythmia Database — [download from Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

- 48 half-hour ambulatory ECG recordings from Beth Israel Hospital
- Independently annotated by two cardiologists
- 87,554 training samples / 21,892 test samples
- 187 samples per heartbeat window at 125 Hz
- 5 classes following AAMI EC57 standard: N, S, V, F, Q

Dataset is not included in this repository due to the PhysioNet licence.

---

## How to Run

**Step 1 — Install dependencies**

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn imbalanced-learn scipy pdfplumber opencv-python-headless Pillow
```

Or on Google Colab:

```python
!pip install pdfplumber opencv-python-headless imbalanced-learn -q
```

**Step 2 — Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) and place `mitbih_train.csv` and `mitbih_test.csv` in your working directory.

**Step 3 — Train the model**

Run the training notebook. The model trains in approximately 8 minutes on a Colab T4 GPU and achieves 98.44% test accuracy.

**Step 4 — Test on your Samsung Health ECG PDF**

Update the PDF path in Cell 1 and run both cells:

```python
PDF_PATH = '/content/your_ecg_export.pdf'
```

Cell 1 reconstructs the signal and detects R-peaks.  
Cell 2 classifies each beat and outputs the diagnosis.

---

## Future Work: ASIC Implementation

The next step is implementing this system as a custom ASIC in Verilog targeting **Skywater 130nm** via the Google Efabless OpenMPW shuttle (free academic fabrication).

**Five chip blocks:**

1. SPI ADC interface — receives samples from AD8232 ECG sensor at 125 Hz
2. IIR bandpass filter — Direct Form II fixed-point, 0.5 to 40 Hz
3. R-peak detector — threshold comparator with cooldown counter
4. CNN accelerator — MAC units, 307,333 INT8 weights in on-chip SRAM
5. Argmax output — drives class and confidence to GPIO and UART

**Why ASIC:**

- Milliwatt power vs watts on a laptop — enables continuous wearable monitoring
- Sub-millisecond latency per beat
- No OS, no Python runtime, no external compute

**Roadmap:**

- Phase 1 (Weeks 1–4): INT8 quantisation via TFLite — 1.2 MB → 300 KB
- Phase 2 (Weeks 5–12): RTL design and unit verification in VCS + DVE
- Phase 3 (Weeks 13–16): Full system integration and simulation
- Phase 4 (Weeks 17–24): Synthesis via OpenLane, physical design, tape-out

---

## Tools Used

Python · TensorFlow · NumPy · SciPy · OpenCV · pdfplumber · Scikit-learn · imbalanced-learn · Matplotlib · Google Colab T4 GPU

---

## Author

**Charan Gundepinni**  
M.S. Electrical Engineering — Colorado State University  
Signal Processing and AI — April 2026  
LinkedIn: [linkedin.com/in/charan-gundepinni-565712249](https://linkedin.com/in/charan-gundepinni-565712249)

---

## Licence

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Dataset:** The MIT-BIH Arrhythmia Database is used under the [PhysioNet Restricted Health Data License](https://physionet.org/content/mitdb/1.0.0/). The dataset is not included in this repository. Download it from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).
