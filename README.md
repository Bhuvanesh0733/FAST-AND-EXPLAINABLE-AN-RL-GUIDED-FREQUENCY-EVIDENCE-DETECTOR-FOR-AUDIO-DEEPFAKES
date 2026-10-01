```
# Fast & Explainable Audio Deepfake Detector
### RL-Guided Frequency Evidence Detection for Synthetic Speech

A lightweight audio deepfake detection system that not only classifies speech as **genuine or synthetic**, but also **explains which frequency bands** revealed the decision — trained with a reinforcement learning policy inspired by GRPO, small enough to run on a laptop CPU.

---

## Overview

| | AASIST | FT-GRPO | **This Project** |
|---|---|---|---|
| Parameters | ~85K | ~7B | ~2M |
| Explainable | ❌ | ✅ | ✅ |
| CPU trainable | ✅ | ❌ | ✅ |
| Evidence type | None | Natural language | Frequency bands |

Most detectors are either fast-but-opaque or interpretable-but-massive. This project targets the gap between them — a three-stage pipeline that is small, fast, and still produces a genuine explanation grounded in the audio signal.

---

## Architecture

```
Raw Audio
    │
    ▼
Preprocessing (16 kHz · mono · 4 s fixed · peak-normalised)
    │
    ├──► Baseline     MFCC features  →  Logistic Regression
    │
    ├──► Stage 1      Wav2Vec2 (frozen) + Attention Pool  →  Binary classifier
    │
    └──► Stage 2      8-band mel energy  →  RL Evidence Policy  →  Verdict + Evidence bands
                                                    │
                                                    ▼
                                            Web Application
                                   (spectrogram overlay · confidence · plain-English explanation)
```

### Stage 2 — RL Evidence Policy (core contribution)

The policy learns **which of 8 perceptually-motivated frequency bands** are sufficient to reach the correct decision. It is trained with a group-relative policy gradient (the same GRPO trick used by FT-GRPO and DeepSeek-R1), without needing a language model:

| Band | Range | Acoustic Meaning |
|---|---|---|
| B1 | 0 – 383 Hz | Fundamental pitch and prosody |
| B2 | 383 – 766 Hz | First vowel formant |
| B3 | 766 – 1166 Hz | Lower-mid vowel structure |
| B4 | 1166 – 1731 Hz | Upper vowel formants, consonant transitions |
| B5 | 1731 – 2569 Hz | Overall speech clarity |
| B6 | 2569 – 3814 Hz | Sibilance and consonant detail |
| B7 | 3814 – 5662 Hz | High-frequency vocoder texture |
| B8 | 5662 – 7999 Hz | Breathiness and fine noise texture |

---

## Results

Evaluated on the **ASVspoof 2019 Logical Access** evaluation set (unseen attacks A07–A19):

| Model | Accuracy | F1 | EER |
|---|---|---|---|
| Baseline (MFCC + LogReg) | 78.31% | 0.860 | 22.74% |
| Stage 1 (Wav2Vec2) | 94.12% | 0.969 | 6.08% |
| Stage 2 (RL Evidence) | 91.27% | 0.946 | 8.91% |

Stage 2 trades ~3% accuracy for a genuine, reward-trained explanation — no hand-coded rules.

---

## Setup

```bash
git clone https://github.com/Bhuvanesh0733/FAST-AND-EXPLAINABLE-AN-RL-GUIDED-FREQUENCY-EVIDENCE-DETECTOR-FOR-AUDIO-DEEPFAKES.git
cd FAST-AND-EXPLAINABLE-AN-RL-GUIDED-FREQUENCY-EVIDENCE-DETECTOR-FOR-AUDIO-DEEPFAKES

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

> Stage 1 downloads `facebook/wav2vec2-base` from Hugging Face on first run — internet required.

---

## Dataset

Download the **ASVspoof 2019 Logical Access** dataset from the University of Edinburgh:
🔗 https://datashare.ed.ac.uk/handle/10283/3336

Download only `LA.zip` (7.12 GB) and unzip into `data/raw/LA/`:

```
data/raw/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

---

## Training

Run the stages in order:

```bash
# 1. Preprocess & build manifests
python -m src.data_prep \
    --la_root data/raw/LA \
    --out_dir data/processed \
    --subsample_train 3000 \
    --subsample_dev 1000

# 2. Baseline (MFCC + Logistic Regression)
python -m src.baseline_model \
    --train data/processed/train_manifest.csv \
    --eval  data/processed/eval_manifest.csv \
    --out_dir checkpoints/baseline

# 3. Stage 1 — Wav2Vec2 transfer learning
python -m src.ssl_classifier train \
    --train data/processed/train_manifest.csv \
    --dev   data/processed/dev_manifest.csv \
    --out_dir checkpoints/stage1 \
    --epochs 10

# 4. Stage 2 — RL evidence policy (CPU, ~minutes)
python -m src.rl_finetune train_classifier \
    --train data/processed/train_manifest.csv \
    --dev   data/processed/dev_manifest.csv \
    --out_dir checkpoints/stage2

python -m src.rl_finetune train_policy \
    --train data/processed/train_manifest.csv \
    --dev   data/processed/dev_manifest.csv \
    --out_dir checkpoints/stage2

# 5. Evaluate all three models
python -m src.evaluate \
    --baseline_metrics checkpoints/baseline/logreg_metrics.json \
    --out_dir reports
```

---

## Running the App

```bash
python app/server.py
```

Open **http://localhost:5000** — upload any `.wav` or `.flac` file and get:
- ✅ Verdict (genuine / synthetic) with confidence
- 📊 Spectrogram with evidence bands highlighted
- 📝 Plain-English explanation of what gave it away

Optionally set `GROQ_API_KEY` in your environment for AI-generated explanations (falls back to rule-based automatically if not set).

---

## Project Structure

```
├── app/
│   ├── server.py          Flask backend
│   ├── templates/         HTML frontend
│   └── static/            CSS + JS
├── src/
│   ├── data_prep.py       Dataset parsing & manifest builder
│   ├── features.py        MFCC, mel-spectrogram, band-energy extraction
│   ├── baseline_model.py  MFCC + LogReg/Random Forest baseline
│   ├── ssl_classifier.py  Wav2Vec2 + attention-pool classifier (Stage 1)
│   ├── rl_finetune.py     GRPO-style RL evidence policy (Stage 2)
│   └── evaluate.py        Metrics table + comparison plots
├── checkpoints/           Saved model weights
├── data/                  Raw & processed dataset
├── reports/               Evaluation outputs
└── requirements.txt
```

---

## References

1. Jung et al., **AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks**, ICASSP 2022. [arXiv:2110.01200](https://arxiv.org/abs/2110.01200)
2. **Interpretable All-Type Audio Deepfake Detection with Audio LLMs via Frequency-Time Reinforcement Learning** [arXiv:2601.02983](https://arxiv.org/abs/2601.02983)
3. Wang et al., **ASVspoof 2019** [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
4. Baevski et al., **wav2vec 2.0**, NeurIPS 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
5. DeepSeek-AI, **DeepSeek-R1** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
```
