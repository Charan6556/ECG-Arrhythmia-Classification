# Academic report status

`SPAI_FINAL_PROJECT_REPORT.pdf` is the original submitted course report. It is retained unchanged as a historical academic artifact.

## Canonical result source

The repository README and `AI_BASED_ARRHYTHMIA_DETECTION_FROM_ECG_SIGNALS.ipynb` reflect the later preserved notebook run and should be used when citing this repository's current results.

| Measurement | Submitted report | Current preserved notebook |
|---|---:|---:|
| Test accuracy | 98.44% | 98.52% |
| Macro F1-score | 0.9153 | 0.9184 |
| Weighted F1-score | 0.9843 | 0.9852 |
| Macro OvR ROC AUC | 0.9925 | 0.9936 |
| Completed training epochs | 46 | 57 |
| Wearable beats classified | 45 | 33 |
| Wearable heart-rate estimate | 93.8 BPM | 68.8 BPM |
| Wearable mean maximum-softmax confidence | 84.3% | 77.1% |

These differences indicate separate experimental runs or pipeline configurations. They must not be combined into one result set.

## Interpretation notice

The report uses terms such as "validation" and "diagnosis" for a single personal wearable example. The current repository treats that example as an exploratory case study only:

- The model predicts heartbeat classes; it does not establish a clinical rhythm diagnosis.
- A single wearable recording is not clinical validation.
- Maximum-softmax probability is not calibrated medical confidence.
- The recorded validation split was created after SMOTE, so validation metrics may be optimistic even though the official test CSV remained untouched.

The report's benchmark and wearable conclusions should therefore be read in the context of a course project, not as evidence of medical-device performance.

## Licensing

The report is original project documentation authored by Charan Gundepinni. Dataset files are not embedded or redistributed. Underlying dataset licensing and attribution are documented in the repository's `DATA_LICENSE.md` file.
