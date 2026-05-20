# ECG Arrhythmia Classification — Cell 2: Classification
# Author: Charan Gundepinni
# M.S. Electrical Engineering, Colorado State University
# April 2026
#
# Requires from Cell 1: ecg_norm, r_peaks, model

from scipy.signal import resample as sp_resample
import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR  = '/content/ecg_images/'
os.makedirs(SAVE_DIR, exist_ok=True)

class_names  = ['N (Normal)', 'S (SVEB)', 'V (VEB)', 'F (Fusion)', 'Q (Unknown)']
class_colors = {0: 'green', 1: 'royalblue', 2: 'red', 3: 'orange', 4: 'purple'}

SEG_LEN   = 187
TARGET_FS = 125

# Calculate RR interval from detected peaks
mean_rr  = int(np.mean(np.diff(r_peaks)))
WIN_SIZE = int(mean_rr * 0.80)   # 80% of RR — one beat only, no overlap
PRE      = WIN_SIZE // 3          # peak sits at 1/3 from left
POST     = WIN_SIZE - PRE - 1

print("=" * 55)
print("   GALAXY WATCH ECG — CLASSIFICATION")
print("=" * 55)
print(f"  Detected peaks   : {len(r_peaks)}")
print(f"  Mean RR interval : {mean_rr} samples ({TARGET_FS*60/mean_rr:.1f} bpm)")
print(f"  Window size      : {WIN_SIZE} samples (80% of RR)")
print(f"  PRE / POST       : {PRE} / {POST}")
print(f"  Resampled to     : {SEG_LEN} samples for model")

# Segment heartbeats
segments    = []
valid_peaks = []

for peak in r_peaks:
    start = peak - PRE
    end   = peak + POST + 1
    if start < 0 or end > len(ecg_norm):
        continue
    seg = ecg_norm[start:end].copy()
    if len(seg) < 10:
        continue
    # Resample short window to 187 to match model input
    seg_resampled = sp_resample(seg, SEG_LEN)
    seg_min = seg_resampled.min()
    seg_max = seg_resampled.max()
    if seg_max - seg_min > 1e-8:
        seg_resampled = (seg_resampled - seg_min) / (seg_max - seg_min)
    segments.append(seg_resampled)
    valid_peaks.append(peak)

segments    = np.array(segments)
valid_peaks = np.array(valid_peaks)
print(f"\n  Valid segments   : {len(segments)}")

# CNN Inference
print("  Running CNN inference...")
X_input = segments.reshape(len(segments), SEG_LEN, 1)
y_prob  = model.predict(X_input, verbose=0)
y_pred  = np.argmax(y_prob, axis=1)
conf    = np.max(y_prob, axis=1)
print(f"  ✓ {len(y_pred)} beats classified")

unique, counts = np.unique(y_pred, return_counts=True)
dominant_class = unique[np.argmax(counts)]

diagnosis_map = {
    0: "Normal Sinus Rhythm",
    1: "Supraventricular Ectopic Beats Detected",
    2: "Ventricular Ectopic Beats Detected",
    3: "Fusion Beats Detected",
    4: "Unknown / Paced Beats Detected"
}
diagnosis    = diagnosis_map[dominant_class]
dominant_bpm = TARGET_FS * 60 / mean_rr

# Per-beat results table
print("\n" + "=" * 45)
print("         PER-BEAT PREDICTIONS")
print("=" * 45)
print(f"{'Beat':>6}  {'Class':<14}  {'Confidence':>10}")
print("-" * 38)
for i, (pred, c) in enumerate(zip(y_pred, conf)):
    flag = "  ⚠" if pred in [2, 3] and c > 0.75 else ""
    print(f"{i+1:>6}  {class_names[pred]:<14}  {c*100:>9.1f}%{flag}")

print("\n" + "=" * 45)
print("         OVERALL SUMMARY")
print("=" * 45)
for u, c_count in zip(unique, counts):
    print(f"  {class_names[u]:<14}: {c_count:>3} beats  ({c_count/len(y_pred)*100:.1f}%)")
print(f"\n  Total beats      : {len(y_pred)}")
print(f"  Detected BPM     : {dominant_bpm:.1f}")
print(f"  Mean confidence  : {conf.mean()*100:.1f}%")
print(f"  High conf >80%   : {(conf > 0.8).sum()} beats")
print(f"  Low  conf <50%   : {(conf < 0.5).sum()} beats")
print(f"\n  Diagnosis        : {diagnosis}")
print("=" * 45)

# FIGURE 1: Annotated ECG
plotted = set()
fig, ax = plt.subplots(figsize=(18, 4))
ax.plot(ecg_norm, color='lightgray', linewidth=0.7, zorder=1)
for peak, pred in zip(valid_peaks, y_pred):
    lbl = class_names[pred] if pred not in plotted else None
    ax.axvline(x=peak, color=class_colors[pred],
               alpha=0.85, linewidth=1.3, label=lbl, zorder=2)
    plotted.add(pred)
ax.set_title(
    f'Galaxy Watch ECG — CNN Per-Beat Classification\n'
    f'BPM: {dominant_bpm:.1f}  |  '
    f'Beats: {len(y_pred)}  |  '
    f'Diagnosis: {diagnosis}',
    fontsize=11
)
ax.set_xlabel('Samples (@125 Hz)'); ax.set_ylabel('Amplitude')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'ecg_classification.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: ecg_classification.png")

# FIGURE 2: First 10 beat windows
n_show = min(10, len(segments))
fig, axes = plt.subplots(2, 5, figsize=(18, 5))
axes = axes.flatten()
for i in range(n_show):
    col = class_colors[y_pred[i]]
    axes[i].plot(segments[i], color=col, linewidth=1.2)
    axes[i].axvline(x=SEG_LEN//3, color='gray',
                    linestyle='--', alpha=0.5, linewidth=0.8)
    axes[i].set_title(
        f'Beat {i+1}\n{class_names[y_pred[i]]}\n{conf[i]*100:.1f}%',
        fontsize=8, color=col
    )
    axes[i].set_ylim(-0.05, 1.05)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xticks([]); axes[i].set_yticks([])
plt.suptitle('First 10 Heartbeat Segments', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'ecg_beat_windows.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: ecg_beat_windows.png")

# FIGURE 3: Confidence distribution + class pie
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(conf*100, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(x=conf.mean()*100, color='red', linestyle='--',
                linewidth=1.5, label=f'Mean: {conf.mean()*100:.1f}%')
axes[0].set_title('Prediction Confidence Distribution')
axes[0].set_xlabel('Confidence (%)'); axes[0].set_ylabel('Number of beats')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

class_counts = [int((y_pred == c).sum()) for c in range(5)]
labels_pie   = [f"{class_names[c]}\n({class_counts[c]})"
                for c in range(5) if class_counts[c] > 0]
sizes_pie    = [class_counts[c] for c in range(5) if class_counts[c] > 0]
colors_pie   = [class_colors[c]  for c in range(5) if class_counts[c] > 0]
axes[1].pie(sizes_pie, labels=labels_pie, colors=colors_pie,
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Class Distribution')

plt.suptitle(
    f'Samsung Galaxy Watch ECG — Results\n'
    f'BPM: {dominant_bpm:.1f}  |  '
    f'Total beats: {len(y_pred)}  |  '
    f'Mean confidence: {conf.mean()*100:.1f}%  |  '
    f'Diagnosis: {diagnosis}',
    fontsize=10
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'ecg_confidence.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Saved: ecg_confidence.png")

print(f"\n{'='*45}")
print(f"  Final Diagnosis: {diagnosis}")
print(f"{'='*45}")
