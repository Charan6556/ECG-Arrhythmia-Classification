# ECG Arrhythmia Classification — Training Script
# Author: Charan Gundepinni
# M.S. Electrical Engineering, Colorado State University
# April 2026

# Install dependencies with: pip install -r requirements.txt

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import kagglehub

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 1. Load Data
path = kagglehub.dataset_download("shayanfazeli/heartbeat")
train_df = pd.read_csv(os.path.join(path, 'mitbih_train.csv'), header=None)
test_df  = pd.read_csv(os.path.join(path, 'mitbih_test.csv'),  header=None)

X_train_raw = train_df.iloc[:, :-1].values
y_train_raw = train_df.iloc[:, -1].values.astype(int)
X_test_raw  = test_df.iloc[:, :-1].values
y_test_raw  = test_df.iloc[:, -1].values.astype(int)

class_names = ['N', 'S', 'V', 'F', 'Q']
num_classes = 5

print('Original train class distribution:')
unique, counts = np.unique(y_train_raw, return_counts=True)
for u, c in zip(unique, counts):
    print(f'  {class_names[u]}: {c}')

# 2. Preprocessing
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=125.0, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def preprocess(X):
    X_f = np.array([bandpass_filter(x) for x in X])
    X_min = X_f.min(axis=1, keepdims=True)
    X_max = X_f.max(axis=1, keepdims=True)
    return (X_f - X_min) / (X_max - X_min + 1e-8)

print('Preprocessing...')
X_train_proc = preprocess(X_train_raw)
X_test_proc  = preprocess(X_test_raw)

# 3. SMOTE
print('Applying SMOTE...')
smote = SMOTE(
    sampling_strategy={
        1: 5000,
        2: 6000,
        3: 3000,
        4: 7000,
    },
    random_state=42,
    k_neighbors=5
)
X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train_raw)

print('Post-SMOTE class distribution:')
unique, counts = np.unique(y_train_sm, return_counts=True)
for u, c in zip(unique, counts):
    print(f'  {class_names[u]}: {c}')

# 4. Train-Test Split + Reshape
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_sm, y_train_sm,
    test_size=0.1, random_state=42, stratify=y_train_sm
)

X_tr_cnn  = X_tr.reshape(X_tr.shape[0],   X_tr.shape[1],  1)
X_val_cnn = X_val.reshape(X_val.shape[0],  X_val.shape[1], 1)
X_te_cnn  = X_test_proc.reshape(X_test_proc.shape[0], X_test_proc.shape[1], 1)

print('Train shape:', X_tr_cnn.shape)

# 5. Focal Loss
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        y_pred    = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce        = -y_true_oh * tf.math.log(y_pred)
        weight    = alpha * y_true_oh * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn

# 6. CNN Architecture
def build_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv1D(64, kernel_size=5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv1D(128, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv1D(256, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 4
    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

model = build_cnn((187, 1), num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)
model.summary()

# 7. Train
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True
)
lr_reduce = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
)

history = model.fit(
    X_tr_cnn, y_tr,
    epochs=80,
    batch_size=128,
    validation_data=(X_val_cnn, y_val),
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# 8. Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Focal Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
plt.tight_layout(); plt.show()

# 9. Evaluation
y_prob = model.predict(X_te_cnn)
y_pred = np.argmax(y_prob, axis=1)

print(f'Test Accuracy : {accuracy_score(y_test_raw, y_pred):.4f}')
print(f'Macro F1      : {f1_score(y_test_raw, y_pred, average="macro"):.4f}')
print(f'Weighted F1   : {f1_score(y_test_raw, y_pred, average="weighted"):.4f}')
print(f'AUC-ROC (OvR) : {roc_auc_score(y_test_raw, y_prob, multi_class="ovr", average="macro"):.4f}')
print('\n--- Classification Report ---')
print(classification_report(y_test_raw, y_pred, target_names=class_names))

# 10. Confusion Matrix
cm = confusion_matrix(y_test_raw, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix — Improved 1D CNN')
plt.tight_layout(); plt.show()

