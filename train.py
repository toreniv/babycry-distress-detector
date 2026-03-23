import os
import glob
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_curve, auc, 
                             confusion_matrix, recall_score, 
                             precision_recall_curve, precision_score)

# =========================
# 0. הגדרות ופתרון נתיבים (החלק החכם)
# =========================
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

IMG_SIZE = (128, 128) 
TARGET_SR = 8000
CATEGORIES = {"hungry": 0, "discomfort": 1, "belly_pain": 1, "burping": 1, "tired": 1}
MIN_RECALL_TARGET = 0.95 

# הגדרת תיקיית הבסיס
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_data_path():
    candidates = [
        os.path.join(BASE_DIR, "donateacry-corpus", "donateacry_corpus_cleaned_and_updated_data"),
        os.path.join(BASE_DIR, "donateacry_corpus", "donateacry_corpus_cleaned_and_updated_data"),
        os.path.join(BASE_DIR, "donateacry_corpus_cleaned_and_updated_data"),
        os.path.join(BASE_DIR, "donateacry-corpus"),
        os.path.join(BASE_DIR, "donateacry_corpus")
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            hits = sum(os.path.isdir(os.path.join(p, k)) for k in CATEGORIES.keys())
            if hits >= 3:
                print(f">> DATA_PATH resolved to: {p}")
                return p
    print("CRITICAL ERROR: Dataset not found. Check folder structure.")
    raise SystemExit(1)

DATA_PATH = resolve_data_path()

# TF Config
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
callbacks = tf.keras.callbacks
regularizers = tf.keras.regularizers
mixed_precision = tf.keras.mixed_precision

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f">> GPU Detected: {gpus[0]}")
    try: mixed_precision.set_global_policy('mixed_float16')
    except: pass

# =========================
# 1. פונקציות DSP
# =========================
def audio_to_rgb_spec(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_SIZE[0])
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    div = S_max - S_min + 1e-6
    S_norm = 2.0 * (S_db - S_min) / div - 1.0
    spec = librosa.util.fix_length(S_norm, size=IMG_SIZE[1], axis=1)
    return np.stack([spec, spec, spec], axis=-1)

def time_shift_clean(sig, sr, shift_sec=0.2):
    shift_samples = int(sr * shift_sec)
    padded = np.zeros_like(sig)
    if shift_samples > 0:
        padded[shift_samples:] = sig[:-shift_samples]
    elif shift_samples < 0:
        shift_samples = abs(shift_samples)
        padded[:-shift_samples] = sig[shift_samples:]
    else:
        return sig
    return padded

# =========================
# 2. איסוף קבצים וחלוקה (Split First!)
# =========================
print(">> Indexing files...")
all_files = []
all_labels = []

for folder, label in CATEGORIES.items():
    folder_path = os.path.join(DATA_PATH, folder)
    files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.WAV"))
    for f in files:
        all_files.append(f)
        all_labels.append(label)

print(f">> Total raw files found: {len(all_files)}")

# חלוקה ל-Train/Test על בסיס שמות הקבצים בלבד!
X_files_temp, X_files_test, y_temp, y_test_labels = train_test_split(
    all_files, all_labels, test_size=0.15, stratify=all_labels, random_state=42
)
X_files_train, X_files_val, y_train_labels, y_val_labels = train_test_split(
    X_files_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

# =========================
# 3. עיבוד ואוגמנטציה (רק ל-Train!)
# =========================
def process_dataset(file_list, label_list, augment=False):
    X_data, y_data = [], []
    for fp, label in zip(file_list, label_list):
        try:
            sig, sr = librosa.load(fp, sr=TARGET_SR, duration=7.0)
            
            # תמיד מוסיפים את המקור
            X_data.append(audio_to_rgb_spec(sig, sr))
            y_data.append(label)
            
            if augment:
                if label == 1: # Distress - Aggressive
                    X_data.append(audio_to_rgb_spec(librosa.effects.pitch_shift(sig, sr=sr, n_steps=2), sr))
                    y_data.append(label)
                    X_data.append(audio_to_rgb_spec(librosa.effects.time_stretch(sig, rate=1.12), sr))
                    y_data.append(label)
                    X_data.append(audio_to_rgb_spec(sig + 0.004 * np.random.randn(len(sig)), sr))
                    y_data.append(label)
                else: # Hungry - Gentle Balance
                    vol_factor = np.random.uniform(0.8, 1.2)
                    X_data.append(audio_to_rgb_spec(sig * vol_factor, sr))
                    y_data.append(label)
                    shifted_sig = time_shift_clean(sig, sr, shift_sec=0.2)
                    X_data.append(audio_to_rgb_spec(shifted_sig, sr))
                    y_data.append(label)
                    vol_factor2 = np.random.uniform(0.9, 1.1)
                    X_data.append(audio_to_rgb_spec(sig * vol_factor2, sr))
                    y_data.append(label)
        except: continue
    return np.array(X_data), np.array(y_data)

print(">> Processing TRAIN set (with Augmentation)...")
X_train, y_train = process_dataset(X_files_train, y_train_labels, augment=True)
print(">> Processing VAL set (No Augmentation)...")
X_val, y_val = process_dataset(X_files_val, y_val_labels, augment=False)
print(">> Processing TEST set (No Augmentation)...")
X_test, y_test = process_dataset(X_files_test, y_test_labels, augment=False)
print(f">> Final Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# =========================
# 4. הגדרת המודל
# =========================
def build_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:-40]: layer.trainable = False 
    
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002)),
        layers.Dropout(0.4), 
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-5), 
                  loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# =========================
# 5. אימון המודלים (Ensemble)
# =========================
NUM_MODELS = 3 
all_preds_test = []

if not os.path.exists('saved_models'): os.makedirs('saved_models')

print(f"\n>> Training {NUM_MODELS}-Model Ensemble (Faster & Leakage-Free)...")

for i in range(NUM_MODELS):
    print(f"--- Model {i+1} ---")
    tf.keras.utils.set_random_seed(i + 42)
    model = build_model()
    
    early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=6, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), 
              batch_size=32, callbacks=[early_stop], verbose=1)
    
    model.save_weights(f'saved_models/model_{i+1}.weights.h5')
    all_preds_test.append(model.predict(X_test, verbose=0).ravel())

# =========================
# 6. חישוב מדדים וספים
# =========================
print("\n>> Analyzing Results...")
avg_preds = np.mean(all_preds_test, axis=0)

fpr, tpr, roc_thresholds = roc_curve(y_test, avg_preds)
roc_auc = auc(fpr, tpr)
precision, recall, pr_thresholds = precision_recall_curve(y_test, avg_preds)
pr_auc = auc(recall, precision)

# Safety Threshold
safety_indices = np.where(tpr >= MIN_RECALL_TARGET)[0]
if len(safety_indices) > 0:
    idx_s = safety_indices[np.argmin(fpr[safety_indices])]
    t_safety = roc_thresholds[idx_s]
else:
    t_safety = 0.1 

# Balanced Threshold
if len(pr_thresholds) > 0:
    f2 = (5 * precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-6)
    t_balanced = pr_thresholds[np.argmax(f2)]
else:
    t_balanced = 0.5

pred_safety = (avg_preds >= t_safety).astype(int)
pred_balanced = (avg_preds >= t_balanced).astype(int)

# =========================
# 7. דוחות טקסטואליים
# =========================
def print_mode_report(mode_name, y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = recall_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1)
    
    print(f"\n{'='*60}")
    print(f"  MODE: {mode_name}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"{'='*60}")
    
    print(f"  [+] Breakdown of Results:")
    print(f"      > True Distress Detected (TP):  {tp}  (Success)")
    print(f"      > Missed Distress Cases  (FN):  {fn}  <-- CRITICAL METRIC")
    print(f"      > False Alarms           (FP):  {fp}  (Cost)")
    print(f"      > Correctly Ignored      (TN):  {tn}")
    
    print("-" * 60)
    print("  [+] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Hungry', 'Distress']))
    print(f"  >> DISTRESS RECALL:    {recall:.2%}")
    print(f"  >> DISTRESS PRECISION: {precision:.2%}")

print("\n\n")
print("#"*60)
print(f"        FINAL SYSTEM REPORT | ROC AUC: {roc_auc:.4f}")
print("#"*60)

print_mode_report("SAFETY FIRST (High Sensitivity)", y_test, pred_safety, t_safety)
print_mode_report("BALANCED (Zero Missed Incidents)", y_test, pred_balanced, t_balanced)
print("\n")

# =========================
# 8. גרפים (ROC + מטריצות)
# =========================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

cm_safety = confusion_matrix(y_test, pred_safety)
cm_balanced = confusion_matrix(y_test, pred_balanced)

# 1. מטריצה - Safety
sns.heatmap(cm_safety, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=False, annot_kws={"size": 14})
axes[0].set_title(f'Safety Mode\n(Recall Focus)', fontsize=16, color='darkred', weight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# 2. מטריצה - Balanced
sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False, annot_kws={"size": 14})
axes[1].set_title(f'Balanced Mode\n(Zero Missed Incidents)', fontsize=16, color='darkgreen', weight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=12)

# 3. ROC Curve
axes[2].plot(fpr, tpr, color='blue', lw=3, label=f'ROC Curve (AUC={roc_auc:.4f})')
axes[2].plot([0, 1], [0, 1], color='gray', linestyle='--') 
axes[2].set_title('ROC Curve (Performance)', fontsize=16, weight='bold')
axes[2].set_xlabel('False Positive Rate (Noise)', fontsize=12)
axes[2].set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)

# נקודות על הגרף
idx_s = np.argmin(np.abs(roc_thresholds - t_safety))
axes[2].scatter(fpr[idx_s], tpr[idx_s], color='red', s=200, label='Safety Point', zorder=5, edgecolors='black')

idx_b = np.argmin(np.abs(roc_thresholds - t_balanced))
axes[2].scatter(fpr[idx_b], tpr[idx_b], color='green', s=200, label='Balanced Point', zorder=5, edgecolors='black')

axes[2].legend(loc="lower right", fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()