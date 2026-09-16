# Dataset licensing and attribution

This repository does not redistribute ECG dataset files.

## Original MIT-BIH Arrhythmia Database

The original waveform records are provided by PhysioNet under the **Open Data Commons Attribution License v1.0 (ODC-By 1.0)**.

- Dataset: https://physionet.org/content/mitdb/1.0.0/
- License: https://opendatacommons.org/licenses/by/1-0/
- DOI: https://doi.org/10.13026/C2F305

Users are responsible for complying with the attribution requirements shown on the official PhysioNet dataset page.

Recommended database reference:

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine. 2001;20(3):45–50.

PhysioNet also requests that users cite its current standard reference listed on the dataset page.

## Preprocessed Kaggle derivative

The training code downloads the **ECG Heartbeat Categorization Dataset** published by Shayan Fazeli on Kaggle:

- https://www.kaggle.com/datasets/shayanfazeli/heartbeat

This derivative contains resampled, segmented CSV data. Access and use are subject to the terms displayed on the Kaggle dataset page. The derivative should be cited separately from the original PhysioNet database.

Associated publication:

> Kachuee M, Fazeli S, Sarrafzadeh M. ECG heartbeat classification: A deep transferable representation. 2018 IEEE International Conference on Healthcare Informatics. 2018.

## Project license boundary

The repository's [MIT License](LICENSE) covers the original source code and documentation in this repository. It does **not** relicense the PhysioNet database, the Kaggle derivative, Samsung Health exports, or any third-party software dependency.
