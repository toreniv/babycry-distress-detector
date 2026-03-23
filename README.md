# Baby Cry Classifier — Hungry vs. Distress

A binary audio classifier that detects whether a baby's cry indicates
**hunger** or **distress** (discomfort, belly pain, burping, tiredness).

Built with **MobileNetV2** transfer learning on **Mel-Spectrograms**,
trained as a **3-model ensemble** with a dual-threshold safety system.

> **Disclaimer:** This project is for research and educational purposes only.
> It is not a medical device and should not be used as a substitute
> for professional medical advice or parental judgment.

---

## Results

| Mode         | Recall (Distress) | Precision | Goal                      |
|--------------|-------------------|-----------|---------------------------|
| Safety First | ≥ 95%             | Lower     | Never miss a distress cry |
| Balanced     | High              | Higher    | F2-optimized trade-off    |

---

## Architecture

```
Raw Audio (.wav)
     ↓
Mel-Spectrogram  (128×128, normalized to [-1, 1])
     ↓
3-channel stack  (RGB format for MobileNetV2)
     ↓
MobileNetV2  (ImageNet weights, top-40 layers unfrozen)
     ↓
GlobalAveragePooling → BatchNorm → Dense(128, ReLU) → Dropout(0.4)
     ↓
Sigmoid output → Dual threshold (Safety / Balanced)
```

**Augmentation strategy:**
- **Distress** (label 1): pitch shift, time stretch, additive noise
- **Hungry** (label 0): volume gain, time shift — gentle balance

**Data split:** 70% Train / 15% Val / 15% Test
Split is performed *before* augmentation to prevent data leakage.

---

## Dataset

This project uses the **Donate-a-Cry Corpus**.
The dataset is **not included** in this repository and must be downloaded separately.

### Option A — GitHub (recommended)

```bash
git clone https://github.com/gveres/donateacry-corpus.git
```

Place the cloned `donateacry-corpus/` folder next to `train.py`.

### Option B — Kaggle (manual download)

1. Go to: https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus
2. Click **Download** and extract to `infant_cry_audio_corpus/`

### Expected folder structure

```
Baby cry/
├── train.py
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
└── donateacry-corpus/
    ├── hungry/          ← label 0  (Hungry)
    ├── discomfort/      ← label 1  (Distress)
    ├── belly_pain/      ← label 1  (Distress)
    ├── burping/         ← label 1  (Distress)
    └── tired/           ← label 1  (Distress)
```

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/baby-cry-classifier.git
cd baby-cry-classifier
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv tensorflow_env
.\tensorflow_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

> If PowerShell blocks script execution:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verify installation

```powershell
python -c "import numpy, tensorflow, librosa; print('All good!')"
```

---

## Run

```powershell
python train.py
```

**Expected output:**
- Training logs for each of the 3 ensemble models
- Classification report (precision / recall / F1 / support)
- ROC-AUC and PR-AUC scores
- `results.png` — confusion matrices + ROC curve (auto-saved)

---

## Project Structure

```
Baby cry/
├── train.py              # Main training + evaluation script
├── requirements.txt      # Python dependencies
├── .gitignore
├── README.md
├── LICENSE
├── saved_models/         # Auto-created at runtime — model weights
└── results.png           # Auto-created at runtime — evaluation plots
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Disable **App Execution Aliases** in Windows Settings → Apps → Advanced app settings |
| `requirements.txt` not found | Make sure you're inside the project folder: `cd "Baby cry"` |
| `ModuleNotFoundError` | Activate the venv first: `.\tensorflow_env\Scripts\Activate.ps1` |
| Pylance red underlines in VSCode | `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `tensorflow_env` |
| Dataset not found | Run `git clone https://github.com/gveres/donateacry-corpus.git` inside the project folder |
| `pip` upgrade notice | Run `python -m pip install --upgrade pip` — optional but recommended |

---

## License

The source code in this repository is licensed under the [MIT License](LICENSE).

The datasets used for training are **not included** in this repository
and must be downloaded separately:

- **Donate-a-Cry Corpus** — licensed under
  [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  Credit: Várallyay György — [github.com/gveres/donateacry-corpus](https://github.com/gveres/donateacry-corpus)

- **Infant Cry Audio Corpus** — licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Source: [kaggle.com/datasets/warcoder/infant-cry-audio-corpus](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)

---

## References

**Papers**
- Abdulaziz & Islam — Infant cry recognition, ICASSP 1995
- Aggarwal et al. — Journal of Voice, 2023 — [doi:10.1016/j.jvoice.2023.05.021](https://www.sciencedirect.com/article/pii/S0892199723001881)
- Frontiers in AI, 2024 — [doi:10.3389/frai.2024.1337356](https://www.frontiersin.org/articles/10.3389/frai.2024.1337356/full)

**Libraries**
- [TensorFlow](https://www.tensorflow.org/) / [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
- [librosa](https://librosa.org/)
- [scikit-learn](https://scikit-learn.org/)
- [seaborn](https://seaborn.pydata.org/) / [matplotlib](https://matplotlib.org/)

---

*Created: January 2026*
