# ECG Arrhythmia Classification — Cell 1: ECG Reconstruction and R-Peak Detection
# Author: Charan Gundepinni
# M.S. Electrical Engineering, Colorado State University
# April 2026

!pip install pdfplumber opencv-python-headless scipy -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber
import cv2
from scipy.signal import butter, filtfilt, resample, find_peaks
from PIL import Image
import io

PDF_PATH = '/content/TEST3.pdf'

# 1. Extract Page Image
with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[0]
    img  = page.to_image(resolution=300)
    buf  = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    page_img = np.array(Image.open(buf).convert('RGB'))

print('Page image shape:', page_img.shape)

# 2. HSV Mask for Orange Trace
hsv  = cv2.cvtColor(page_img, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([25, 255, 255]))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 3. Detect Exactly 3 ECG Strips (skip calibration pulse)
H, W        = mask.shape
row_sums    = mask.sum(axis=1)
active_rows = np.where(row_sums > W * 0.02)[0]

strip_bounds = []
if len(active_rows) > 0:
    gaps   = np.where(np.diff(active_rows) > 30)[0]
    starts = np.concatenate([[active_rows[0]], active_rows[gaps + 1]])
    ends   = np.concatenate([active_rows[gaps], [active_rows[-1]]])
    for s, e in zip(starts, ends):
        if e - s > 100:   # skip calibration pulse
            strip_bounds.append((s, e))

print(f'Detected {len(strip_bounds)} ECG strips: {strip_bounds}')
assert len(strip_bounds) == 3, f'Expected 3 strips, got {len(strip_bounds)} — tune threshold'

# 4. Extract + Resample Each Strip to 1250 Samples (10s @ 125Hz)
def extract_strip_signal(mask_strip):
    signal = []
    for col in range(mask_strip.shape[1]):
        rows = np.where(mask_strip[:, col] > 0)[0]
        signal.append(np.mean(rows) if len(rows) > 0 else np.nan)
    signal = np.array(signal)
    nans = np.isnan(signal)
    if nans.any():
        x = np.arange(len(signal))
        signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    return -signal  # invert: high pixel row = low voltage

TARGET_FS         = 125
SAMPLES_PER_STRIP = TARGET_FS * 10   # 1250 samples per 10s strip

all_strips = []
for i, (r0, r1) in enumerate(strip_bounds):
    sig = extract_strip_signal(mask[r0:r1, :])
    sig_resampled = resample(sig, SAMPLES_PER_STRIP)
    all_strips.append(sig_resampled)
    plt.figure(figsize=(14, 2))
    plt.plot(sig_resampled, color='darkorange', linewidth=0.8)
    plt.title(f'Strip {i+1} — Resampled to {SAMPLES_PER_STRIP} samples')
    plt.xlabel('Samples @125Hz'); plt.tight_layout(); plt.show()

ecg_125hz = np.concatenate(all_strips)
print(f'Total signal: {len(ecg_125hz)} samples @ {TARGET_FS} Hz (expected {30*TARGET_FS})')

# 5. Bandpass Filter + Normalize
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=125.0, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

ecg_filtered = bandpass_filter(ecg_125hz, fs=TARGET_FS)
ecg_norm     = (ecg_filtered - ecg_filtered.min()) / (ecg_filtered.max() - ecg_filtered.min() + 1e-8)

plt.figure(figsize=(16, 3))
plt.plot(ecg_norm, color='steelblue', linewidth=0.7)
plt.title('Reconstructed ECG — Filtered & Normalized (125 Hz)')
plt.xlabel('Samples'); plt.ylabel('Amplitude')
plt.tight_layout(); plt.show()

# 6. R-Peak Detection
r_peaks, _ = find_peaks(ecg_norm, height=0.45, distance=45, prominence=0.2)
print(f'R-peaks detected : {len(r_peaks)}')
print(f'Expected (~108 bpm × 30s) : ~{int(108*30/60)} beats')

plt.figure(figsize=(16, 3))
plt.plot(ecg_norm, color='steelblue', linewidth=0.7, label='ECG')
plt.plot(r_peaks, ecg_norm[r_peaks], 'rv', markersize=8, label='R-peaks')
plt.title('R-Peak Detection'); plt.xlabel('Samples')
plt.legend(); plt.tight_layout(); plt.show()

# 7. Segment Heartbeats
SEGMENT_LEN = 187
half        = SEGMENT_LEN // 2
segments, valid_peaks = [], []

for peak in r_peaks:
    start = peak - half
    end   = start + SEGMENT_LEN
    if start >= 0 and end <= len(ecg_norm):
        seg = ecg_norm[start:end]
        seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-8)
        segments.append(seg)
        valid_peaks.append(peak)

segments    = np.array(segments)
valid_peaks = np.array(valid_peaks)
print(f'Valid segments: {len(segments)}')
