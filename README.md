# Fast & Explainable Audio Deepfake Detector

### RL-Guided Frequency Evidence Detection for Synthetic Speech

A lightweight audio deepfake detection system that not only classifies speech as **genuine or synthetic**, but also explains which frequency bands contributed to the decision. The system uses a reinforcement learning policy inspired by GRPO and is designed to be small enough to train and run on a laptop CPU.

---

## Overview

|               | AASIST |          FT-GRPO | **This Project** |
| ------------- | -----: | ---------------: | ---------------: |
| Parameters    |   ~85K |              ~7B |              ~2M |
| Explainable   |     No |              Yes |              Yes |
| CPU Trainable |    Yes |               No |              Yes |
| Evidence Type |   None | Natural Language |  Frequency Bands |

Most audio deepfake detectors are either fast but difficult to interpret or interpretable but computationally expensive. This project addresses that gap through a three-stage pipeline that is lightweight, efficient, and capable of producing explanations grounded in the audio signal.

---

## Architecture

```text
Raw Audio
    |
    v
Preprocessing
(16 kHz, mono, 4 s fixed, peak-normalised)
    |
    +-------------------> Baseline
    |                      MFCC Features
    |                          |
    |                          v
    |                   Logistic Regression
    |
    +-------------------> Stage 1
    |                      Wav2Vec2 (frozen)
    |                          |
    |                     Attention Pool
    |                          |
    |                          v
    |                    Binary Classifier
    |
    +-------------------> Stage 2
                           8-Band Mel Energy
                                  |
                                  v
                         RL Evidence Policy
                                  |
                                  v
                         Verdict + Evidence
                               Bands
                                  |
                                  v
                          Web Application
                    Spectrogram Overlay
                    Confidence Score
                    Plain-English Explanation
```

---

## Stage 2: RL Evidence Policy

The core contribution of the project is the reinforcement learning-based evidence policy.

The policy learns which of eight perceptually motivated frequency bands are sufficient to support the correct classification. It uses a group-relative policy gradient approach inspired by GRPO, similar to the optimization strategy used in FT-GRPO and DeepSeek-R1, but without requiring a language model.

| Band | Frequency Range | Acoustic Meaning                               |
| ---- | --------------- | ---------------------------------------------- |
| B1   | 0 - 383 Hz      | Fundamental pitch and prosody                  |
| B2   | 383 - 766 Hz    | First vowel formant                            |
| B3   | 766 - 1166 Hz   | Lower-mid vowel structure                      |
| B4   | 1166 - 1731 Hz  | Upper vowel formants and consonant transitions |
| B5   | 1731 - 2569 Hz  | Overall speech clarity                         |
| B6   | 2569 - 3814 Hz  | Sibilance and consonant detail                 |
| B7   | 3814 - 5662 Hz  | High-frequency vocoder texture                 |
| B8   | 5662 - 7999 Hz  | Breathiness and fine noise texture             |

The selected bands serve as evidence for the model's final prediction and can be visualised directly on the spectrogram.

---

## Results

The system was evaluated on the **ASVspoof 2019 Logical Access evaluation set**, containing unseen attacks A07-A19.

| Model                                 | Accuracy | F1 Score |    EER |
| ------------------------------------- | -------: | -------: | -----: |
| Baseline (MFCC + Logistic Regression) |   78.31% |    0.860 | 22.74% |
| Stage 1 (Wav2Vec2)                    |   94.12% |    0.969 |  6.08% |
| Stage 2 (RL Evidence)                 |   91.27% |    0.946 |  8.91% |

Stage 2 provides a trade-off between classification performance and interpretability. It achieves 91.27% accuracy while producing reward-trained frequency evidence rather than relying entirely on manually defined explanation rules.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Bhuvanesh0733/FAST-AND-EXPLAINABLE-AN-RL-GUIDED-FREQUENCY-EVIDENCE-DETECTOR-FOR-AUDIO-DEEPFAKES.git

cd FAST-AND-EXPLAINABLE-AN-RL-GUIDED-FREQUENCY-EVIDENCE-DETECTOR-FOR-AUDIO-DEEPFAKES
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Stage 1 downloads `facebook/wav2vec2-base` from Hugging Face during the first run, so an internet connection is required.

---

## Dataset

This project uses the **ASVspoof 2019 Logical Access** dataset.

Dataset source:

https://datashare.ed.ac.uk/handle/10283/3336

Download the `LA.zip` archive and extract it into:

```text
data/raw/LA/
```

The expected directory structure is:

```text
data/raw/LA/
├── ASVspoof2019_LA_train/
│   └── flac/
├── ASVspoof2019_LA_dev/
│   └── flac/
├── ASVspoof2019_LA_eval/
│   └── flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

---

## Training

Run the following stages in order.

### 1. Preprocess and Build Manifests

```bash
python -m src.data_prep \
    --la_root data/raw/LA \
    --out_dir data/processed \
    --subsample_train 3000 \
    --subsample_dev 1000
```

This parses the ASVspoof dataset and creates the training, development, and evaluation manifests used by the subsequent stages.

### 2. Train the Baseline

The baseline uses MFCC features with Logistic Regression.

```bash
python -m src.baseline_model \
    --train data/processed/train_manifest.csv \
    --eval data/processed/eval_manifest.csv \
    --out_dir checkpoints/baseline
```

### 3. Train Stage 1

Stage 1 uses a pretrained Wav2Vec2 model with an attention-pooling classification head.

```bash
python -m src.ssl_classifier train \
    --train data/processed/train_manifest.csv \
    --dev data/processed/dev_manifest.csv \
    --out_dir checkpoints/stage1 \
    --epochs 10
```

### 4. Train the Stage 2 RL Classifier

```bash
python -m src.rl_finetune train_classifier \
    --train data/processed/train_manifest.csv \
    --dev data/processed/dev_manifest.csv \
    --out_dir checkpoints/stage2
```

### 5. Train the RL Evidence Policy

```bash
python -m src.rl_finetune train_policy \
    --train data/processed/train_manifest.csv \
    --dev data/processed/dev_manifest.csv \
    --out_dir checkpoints/stage2
```

### 6. Evaluate the Models

```bash
python -m src.evaluate \
    --baseline_metrics checkpoints/baseline/logreg_metrics.json \
    --out_dir reports
```

---

## Running the Web Application

Start the Flask server:

```bash
python app/server.py
```

Open the following address in a browser:

```text
http://localhost:5000
```

The application accepts `.wav` and `.flac` audio files and provides:

* Genuine or synthetic classification
* Prediction confidence
* Spectrogram visualisation
* Highlighted frequency evidence bands
* Plain-English explanation of the detected evidence

---

## AI-Generated Explanations

The application can optionally use the Groq API to generate natural-language explanations.

Set the `GROQ_API_KEY` environment variable before starting the application.

If the API key is not configured, the application automatically falls back to a rule-based explanation system.

For Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key"
```

For Linux/macOS:

```bash
export GROQ_API_KEY="your_api_key"
```

---

## Project Structure

```text
FAST-AND-EXPLAINABLE-AN-RL-GUIDED-FREQUENCY-EVIDENCE-DETECTOR-FOR-AUDIO-DEEPFAKES/
│
├── app/
│   ├── server.py
│   ├── templates/
│   │   └── HTML frontend
│   └── static/
│       ├── CSS
│       └── JavaScript
│
├── src/
│   ├── data_prep.py
│   │   └── Dataset parsing and manifest generation
│   │
│   ├── features.py
│   │   └── MFCC, mel-spectrogram and band-energy extraction
│   │
│   ├── baseline_model.py
│   │   └── MFCC + Logistic Regression baseline
│   │
│   ├── ssl_classifier.py
│   │   └── Wav2Vec2 + attention-pooling classifier
│   │
│   ├── rl_finetune.py
│   │   └── GRPO-style RL evidence policy
│   │
│   └── evaluate.py
│       └── Metrics and model comparison
│
├── checkpoints/
│   ├── baseline/
│   ├── stage1/
│   └── stage2/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   └── Evaluation outputs
│
├── requirements.txt
└── README.md
```

---

## Methodology

The system consists of three progressively more advanced detection approaches.

### Baseline

The baseline extracts Mel-Frequency Cepstral Coefficients from the input speech and uses Logistic Regression for binary classification.

```text
Audio
  |
  v
MFCC Extraction
  |
  v
Logistic Regression
  |
  v
Genuine / Synthetic
```

### Stage 1

Stage 1 uses a pretrained Wav2Vec2 representation extractor. The pretrained feature encoder remains frozen while an attention-pooling classification layer learns to distinguish genuine and synthetic speech.

```text
Audio
  |
  v
Wav2Vec2
  |
  v
Attention Pooling
  |
  v
Binary Classifier
  |
  v
Genuine / Synthetic
```

### Stage 2

Stage 2 introduces the reinforcement learning evidence policy.

The audio is converted into eight frequency-band energy representations. The policy selects informative bands and receives rewards based on classification correctness and evidence quality.

```text
Audio
  |
  v
Mel Spectrogram
  |
  v
8 Frequency Bands
  |
  v
RL Evidence Policy
  |
  +--------------------+
  |                    |
  v                    v
Selected Evidence    Classification
  |                    |
  +---------+----------+
            |
            v
     Final Explanation
```

---

## Explainability

Unlike a conventional classifier that only produces a class probability, this system identifies the frequency regions that contributed to its decision.

For example, the system may identify high-frequency regions such as B7 or B8 as important evidence. These regions can then be highlighted on the spectrogram shown in the web interface.

This provides a direct connection between:

```text
Audio Signal
     |
     v
Frequency Representation
     |
     v
Selected Evidence Bands
     |
     v
Classification Decision
     |
     v
Human-Readable Explanation
```

---

## Computational Design

The project is designed with lightweight deployment in mind.

| Component               | Design Choice                  |
| ----------------------- | ------------------------------ |
| Audio Sampling Rate     | 16 kHz                         |
| Audio Channels          | Mono                           |
| Input Duration          | 4 seconds                      |
| Baseline                | MFCC + Logistic Regression     |
| SSL Model               | Wav2Vec2                       |
| Evidence Representation | 8 Mel Frequency Bands          |
| RL Method               | Group-Relative Policy Gradient |
| Backend                 | Flask                          |
| Deployment Target       | Laptop CPU                     |

The RL evidence policy is intentionally designed without requiring a large language model, reducing computational requirements while retaining an interpretable evidence-selection mechanism.

---

## Limitations

The current implementation has several limitations:

1. The evaluation is based on the ASVspoof 2019 Logical Access dataset and may not represent every modern voice-conversion or text-to-speech system.

2. The Stage 2 model has lower classification performance than the Stage 1 Wav2Vec2 model.

3. Frequency-band evidence provides an acoustic interpretation but does not necessarily identify the exact generative mechanism responsible for a synthetic recording.

4. The current preprocessing uses a fixed four-second audio segment.

5. The Wav2Vec2 stage requires downloading the pretrained model before first use.

6. Generalisation to unseen datasets and real-world recording conditions requires additional evaluation.

---

## References

1. Jung et al., **AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks**, ICASSP 2022.
   https://arxiv.org/abs/2110.01200

2. **Interpretable All-Type Audio Deepfake Detection with Audio LLMs via Frequency-Time Reinforcement Learning**.
   https://arxiv.org/abs/2601.02983

3. Wang et al., **ASVspoof 2019: A Large-Scale Public Database of Synthetic, Converted and Replay Speech**, Computer Speech & Language.
   https://arxiv.org/abs/1911.01601

4. Baevski et al., **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**, NeurIPS 2020.
   https://arxiv.org/abs/2006.11477

5. DeepSeek-AI, **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**.
   https://arxiv.org/abs/2501.12948

---

## License

This project is intended for research and educational purposes.

---

## Author

**Bhuvanesh A**

GitHub:
https://github.com/Bhuvanesh0733
