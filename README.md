# EMG Gesture Classification — NinaPro DB5

Classification of hand and wrist gestures from surface EMG signals using the [NinaPro DB5](http://ninapro.hevs.ch/) dataset.

## Motivation

Gesture recognition from EMG has direct applications in prosthetic limb control, human-robot interaction, and rehabilitation. This project implements a complete signal processing and machine learning pipeline — from raw EMG to trained classifiers — using classical time/frequency-domain features.

## Dataset

**NinaPro DB5** — 10 intact subjects, 53 hand gestures, 16 sEMG channels (Myo armbands), 200 Hz sampling rate, 6 repetitions per gesture.

Download: [ninapro.hevs.ch](http://ninapro.hevs.ch/) (registration required)

## Pipeline

```
Raw EMG (200 Hz, 16ch)
      │
      ▼
Bandpass filter (20–490 Hz) + Notch (60 Hz)
      │
      ▼
Sliding window segmentation (200 ms, 10 ms step)
      │
      ▼
Feature extraction (MAV, RMS, ZCR, SSC, WL, Hjorth, Mean Freq)
      │         → 9 features × 16 channels = 144-dim vector
      ▼
Classifiers: SVM · Random Forest · XGBoost
      │
      ▼
Evaluation: Accuracy · F1 macro · Confusion matrix
```

**Train/Val/Test split:** by repetition — reps 1–4 / rep 5 / rep 6. No data leakage between splits.

## Results

| Classifier     | Val Accuracy | Test Accuracy | Test F1 (macro) |
|----------------|:------------:|:-------------:|:---------------:|
| SVM (RBF)      | 0.546        | 0.558         | 0.561           |
| Random Forest  | 0.674        | 0.676         | 0.676           |
| XGBoost        | 0.640        | 0.641         | 0.641           |

<img width="1350" height="750" alt="classifier_comparison" src="https://github.com/user-attachments/assets/e04e067e-33ce-428b-9b63-e151d04eaf95" />

### Confusion Matrix

SVM
<img width="1800" height="1500" alt="SVM_RBF_cm" src="https://github.com/user-attachments/assets/6bfdfb18-bb32-4db6-a0ec-cb1481eeb1f8" />

Random Forest
<img width="1800" height="1500" alt="Random_Forest_cm" src="https://github.com/user-attachments/assets/ee884ff7-f968-4151-8827-b42ba9c5c9d7" />

XGBoost
<img width="1800" height="1500" alt="XGBoost_cm" src="https://github.com/user-attachments/assets/26fbd6bf-77ba-440f-94b1-d41a1d87a331" />

## Project Structure

```
emg-gesture-classification/
├── src/
│   ├── preprocessing.py   # filtering, segmentation, normalization
│   ├── features.py        # time/frequency feature extraction
│   └── train.py           # classifiers, evaluation, plots
├── data/                  # place NinaPro .mat files here
├── results/               # metrics JSON + plots (auto-generated)
├── notebooks/             # exploratory analysis
├── main.py                # entry point
└── requirements.txt
```

## Usage

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place NinaPro DB5 .mat files in /data and run
python main.py --mat_files data/S1_E1_A1.mat data/S1_E2_A1.mat data/S1_E3_A1.mat
```

## Reproducing results

- Window: 200 ms · Step: 10 ms
- Normalization: z-score, statistics computed on training set only
- No inter-subject transfer — each subject trained independently

## References

- Atzori et al. (2015). *Electromyography data for non-invasive naturally-controlled robotic hand prostheses.* Scientific Data.
- Hudgins et al. (1993). *A new strategy for multifunction myoelectric control.* IEEE TNSRE.
