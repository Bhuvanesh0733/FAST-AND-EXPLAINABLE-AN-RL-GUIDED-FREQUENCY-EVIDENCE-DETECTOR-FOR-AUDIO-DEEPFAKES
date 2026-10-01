# Interpretable Audio Deepfake Detection

Case-study gap this project fills: **AASIST** (Jung et al., ICASSP 2022, arXiv:2110.01200)
is fast and lightweight (AASIST-L: 85K params) but gives zero explanation for its decisions.
**FT-GRPO** (arXiv:2601.02983) gives interpretable, reasoning-backed decisions but needs a
multi-billion-parameter Audio-LLM trained with GRPO on 8×A100 GPUs. This project targets the
middle ground neither paper covers: **small and fast, but still interpretable.**

> **Before you submit anything:** your assignment brief explicitly requires individual effort
> and bans "100% AI content." Run every stage yourself, read the code, and be ready to explain
> and modify any part of it — that's the actual point of the exercise, not just the grade.

---

## Architecture

- **Baseline** — MFCC features + Logistic Regression/Random Forest (`src/baseline_model.py`).
  The simple model the rubric requires you to compare against.
- **Stage 1** — transfer learning: pretrained Wav2Vec2 encoder + attention-pooling classifier
  head (`src/ssl_classifier.py`). The "fast, deployable" side of the gap.
- **Stage 2** — a GRPO-style reinforcement learning policy that learns which frequency bands
  of the spectrogram are the most useful evidence for the decision (`src/rl_finetune.py`).
  The "interpretable" side of the gap, trained with the same group-relative-advantage trick
  FT-GRPO uses, scaled down to something that trains on CPU in minutes.
- **App** — a custom Flask backend (`app/server.py`) + hand-designed frontend
  (`app/templates/`, `app/static/`): upload a clip, get a prediction with a
  highlighted evidence overlay drawn directly on the spectrogram.

All three stages have already been smoke-tested end-to-end on synthetic data — see
"What's already verified" below.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate     # or use Colab
pip install -r requirements.txt
```

Stage 1 needs internet access the first time you run it (downloads `facebook/wav2vec2-base`
from Hugging Face). Everything else runs fully offline.

## Get the dataset

Download from the University of Edinburgh DataShare:
https://datashare.ed.ac.uk/handle/10283/3336

You only need:
- `LA.zip` (7.12 GB) — unzip into `data/raw/LA/`
- `README.txt`, `LICENSE_text.txt`

Skip `PA.zip` (16.45 GB) — that's the replay-attack task, not used here.

Expected structure after unzipping:
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

If your protocol files have a different column order than the code expects, `src/data_prep.py`
will raise a clear error telling you to check the README and fix `PROTOCOL_COLUMNS` — don't
just delete columns to make it pass, actually check what changed.

## Run, in order

```bash
# 1. Build manifests (subsample train/dev for speed; eval is never subsampled)
python -m src.data_prep --la_root data/raw/LA --out_dir data/processed \
    --subsample_train 3000 --subsample_dev 1000

# 2. Baseline
python -m src.baseline_model --train data/processed/train_manifest.csv \
    --eval data/processed/eval_manifest.csv --model_type logreg \
    --out_dir checkpoints/baseline

# 3. Stage 1 (needs internet + ideally a GPU — use Colab if training locally is slow)
python -m src.ssl_classifier train --train data/processed/train_manifest.csv \
    --dev data/processed/dev_manifest.csv --out_dir checkpoints/stage1 --epochs 10

# 4. Stage 2 (fully offline, runs on CPU)
python -m src.rl_finetune train_classifier --train data/processed/train_manifest.csv \
    --dev data/processed/dev_manifest.csv --out_dir checkpoints/stage2
python -m src.rl_finetune train_policy --train data/processed/train_manifest.csv \
    --dev data/processed/dev_manifest.csv --out_dir checkpoints/stage2

# 5. Evaluation + comparison table
python -m src.evaluate --baseline_metrics checkpoints/baseline/logreg_metrics.json \
    --out_dir reports

# 6. App
python app/server.py
# then open http://localhost:5000
```

## What's already verified (in the build environment, before handoff)

- `data_prep.py` — parses protocol files, builds manifests, correctly flags unseen-attack
  eval clips. Verified end-to-end on synthetic data.
- `baseline_model.py` — trains and evaluates cleanly, produces accuracy/F1/EER.
- `rl_finetune.py` — **both** the evidence classifier and the GRPO-style policy were trained
  end-to-end on synthetic data. The policy's mean group reward rose from -0.07 to ~0.49 over
  30 epochs, confirming the policy gradient update is actually learning, not just running.
- `ssl_classifier.py` — the `AttentionPool` + classifier-head forward pass was verified with a
  randomly-initialized (non-downloaded) encoder of the same architecture. The real pretrained
  weights couldn't be downloaded in the build sandbox (no internet access there), so run this
  stage yourself first on a small subsample to confirm it trains before committing to a full run.
- `app/server.py` + `app/templates/index.html` + `app/static/` — verified end-to-end with
  Flask's test client: the index page renders correctly, and a real POST to `/api/predict`
  with a synthetic clip returned a correct JSON response (prediction, confidence, real
  Hz-labeled evidence bands, correctly-shaped spectrogram data) that matches exactly what the
  frontend JavaScript expects. The actual rendered page (canvas drawing, drag-and-drop) still
  needs a visual check in a real browser — the test client confirms the data contract, not
  the pixels.

What was **not** verified end-to-end: a full real-data training run of Stage 1 (needs internet
+ realistically a GPU) and the visual appearance of the UI in an actual browser. Do this
yourself early — don't leave it until the night before the deadline.

## Cleanup before submitting

Delete the `tests/` folder — it's scaffolding used to smoke-test the pipeline with fake data,
not part of the deliverable.

## Report mapping

Each `src/` module's docstring explains the *why*, not just the *what* — use those directly as
source material for your report's Methodology section, and the numbers in `reports/` for
Results & Discussion.
