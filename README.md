# ECG Arrhythmia Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A 1D convolutional neural network for five-class heartbeat classification, plus an experimental computer-vision pipeline that reconstructs a single-lead ECG trace from a Samsung Health PDF export.

> **Research-use disclaimer:** This repository is an educational machine-learning project. It is not a medical device and must not be used to diagnose, treat, or rule out a medical condition. The wearable example is a single exploratory case study, not clinical validation.

## Highlights

- Five AAMI-aligned heartbeat categories: N, S, V, F, and Q
- Band-pass filtering and per-beat normalization
- SMOTE augmentation for minority classes
- 1D CNN trained with focal loss
- PDF trace extraction using HSV masking and pixel-centroid tracking
- Per-beat prediction summaries and confidence visualizations

## Repository contents

| File | Purpose |
|---|---|
| `AI_BASED_ARRHYTHMIA_DETECTION_FROM_ECG_SIGNALS.ipynb` | Complete training, PDF reconstruction, and inference workflow |
| `train_model.py` | Model training and benchmark evaluation |
| `cell1_ecg_reconstruction.py` | Samsung Health PDF trace extraction and R-peak detection |
| `cell2_classification.py` | Per-beat classification for a reconstructed ECG |
| `requirements.txt` | Python dependencies |
| `DATA_LICENSE.md` | Dataset provenance, licensing, and citation notes |

## Recorded benchmark results

The following values are copied directly from the saved notebook output for the 21,892-sample test set. They describe one recorded run and may vary slightly when retrained.

| Metric | Recorded value |
|---|---:|
| Test accuracy | 98.52% |
| Macro F1-score | 0.9184 |
| Weighted F1-score | 0.9852 |
| Macro one-vs-rest ROC AUC | 0.9936 |
| Model parameters | 307,333 |
| Completed epochs | 57 of 80 |

### Per-class test performance

| Class | Description | Precision | Recall | F1-score | Support |
|---|---|---:|---:|---:|---:|
| N | Normal beat | 0.99 | 0.99 | 0.99 | 18,118 |
| S | Supraventricular ectopic beat | 0.88 | 0.81 | 0.84 | 556 |
| V | Ventricular ectopic beat | 0.98 | 0.95 | 0.96 | 1,448 |
| F | Fusion beat | 0.73 | 0.89 | 0.80 | 162 |
| Q | Unknown or paced beat | 0.99 | 0.99 | 0.99 | 1,608 |

Accuracy is influenced by the large Normal class. Macro F1 is therefore the more informative single-number summary for minority-class performance.

### Evaluation limitation

In the recorded run, SMOTE was applied before the training/validation split. The official test CSV remained untouched, but the validation score used for early stopping may be optimistic because synthetic neighbors can appear across the training and validation subsets. A stricter follow-up experiment should split first, apply SMOTE only to the training subset, and preferably evaluate with a patient-independent split.

## Wearable PDF case study

The notebook also contains one exploratory Samsung Health PDF example. Its saved output reports:

| Measurement | Recorded value |
|---|---:|
| Detected and classified beats | 33 |
| Estimated heart rate from mean R-R interval | 68.8 BPM |
| N predictions | 32 (97.0%) |
| V predictions | 1 (3.0%) |
| Mean maximum-softmax confidence | 77.1% |
| Predictions above 80% confidence | 12 |

These values are pipeline outputs, not clinically verified labels. A dominant N prediction does not by itself establish normal sinus rhythm, and maximum-softmax probabilities should not be interpreted as calibrated medical confidence.

## Method

### Training pipeline

1. Download the preprocessed MIT-BIH heartbeat CSV files through `kagglehub`.
2. Apply a 0.5–40 Hz Butterworth band-pass filter to each 187-sample beat.
3. Normalize each beat to the `[0, 1]` range.
4. Augment selected minority classes with SMOTE.
5. Train a four-block 1D CNN using focal loss and early stopping.
6. Evaluate once on the untouched test CSV.

### Model architecture

```text
Input (187 × 1)
  → Conv1D(64, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout
  → Conv1D(128, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout
  → Conv1D(256, kernel=3) + BatchNorm + ReLU + MaxPool + Dropout
  → Conv1D(128, kernel=3) + BatchNorm + ReLU + Dropout
  → GlobalAveragePooling1D
  → Dense(256) + Dropout
  → Dense(128) + Dropout
  → Dense(5, softmax)
```

The early convolutional kernels learn short local waveform patterns. Deeper layers and pooling combine those features across a larger effective receptive field.

### PDF reconstruction pipeline

1. Render the first PDF page at 300 DPI.
2. Isolate the orange ECG trace in HSV color space.
3. Detect the three ECG strips using row-density analysis.
4. Track the trace centroid column by column and interpolate gaps.
5. Resample each ten-second strip to 1,250 samples at 125 Hz.
6. Filter the reconstructed signal, detect R-peaks, and classify beat windows.

The extraction thresholds are specific to the tested Samsung Health PDF layout and may require adjustment for other exports or application versions.

## Dataset

Training uses the preprocessed [ECG Heartbeat Categorization Dataset on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat), derived from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

The CSV derivative contains 187 signal samples plus one class label per row:

- Training samples: 87,554
- Test samples: 21,892
- Sampling rate of the derivative: 125 Hz

The original PhysioNet database contains 48 half-hour, two-channel recordings sampled at 360 Hz. The processed CSV representation is therefore not identical to the original waveform files. Dataset files are not redistributed in this repository. See [DATA_LICENSE.md](DATA_LICENSE.md) for licensing and attribution.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/Charan6556/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Usage

### Complete notebook workflow

Open `AI_BASED_ARRHYTHMIA_DETECTION_FROM_ECG_SIGNALS.ipynb` in Jupyter or Google Colab and run the cells in order. Kaggle authentication may be required for the first dataset download.

### Training script

```bash
python train_model.py
```

### Wearable PDF workflow

Set the PDF path with the `ECG_PDF_PATH` environment variable (or edit the default in `cell1_ecg_reconstruction.py`), then execute the reconstruction and classification code in the same Python or notebook session so that the trained `model`, `ecg_norm`, and `r_peaks` objects are available. Generated figures are written to `ecg_images/` by default; set `ECG_OUTPUT_DIR` to change that location.

Do not commit personal ECG exports, patient identifiers, model credentials, or Kaggle API tokens.

## Reproducibility notes

- The uploaded notebook has execution outputs cleared to avoid publishing large embedded images or personal ECG content.
- Numerical results above were transcribed from the preserved local notebook output.
- The model weights and source Samsung Health PDF are not included.
- TensorFlow training can vary across hardware and software versions because all operations are not guaranteed to be deterministic.

## Future work

- Move SMOTE after the training/validation split and rerun all metrics.
- Add patient-independent evaluation using record identifiers.
- Report confidence calibration, confidence intervals, and repeated-seed results.
- Validate the PDF reconstruction pipeline on multiple devices and ECG exports with reference labels.
- Save versioned model weights and machine-readable experiment metadata.

## Citation

When using the underlying MIT-BIH data, cite the database and PhysioNet as requested on the [official dataset page](https://physionet.org/content/mitdb/1.0.0/). The preprocessed CSV dataset is associated with:

> Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). ECG heartbeat classification: A deep transferable representation. *2018 IEEE International Conference on Healthcare Informatics*.

## Author

Charan Gundepinni  
M.S. Electrical Engineering, Colorado State University  
[LinkedIn](https://linkedin.com/in/charan-gundepinni-565712249)

## License

The project code and documentation are released under the [MIT License](LICENSE). Dataset files are not covered by the project license; see [DATA_LICENSE.md](DATA_LICENSE.md).
