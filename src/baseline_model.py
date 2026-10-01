"""
baseline_model.py
------------------
The "simple baseline" required by the rubric (Section 2c: "compare to a
simpler baseline model") — classical ML on handcrafted MFCC features.
No deep learning, no transfer learning, no RL. This exists purely so you
can quantify how much the advanced pipeline (Stage 1 / Stage 2) actually
buys you over a cheap, minutes-to-train alternative.

Usage:
    python -m src.baseline_model --train data/processed/train_manifest.csv \
                                  --eval  data/processed/eval_manifest.csv \
                                  --out_dir checkpoints/baseline
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from .features import extract_mfcc_vector, load_audio


def featurize_manifest(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for path in tqdm(df.file_path, desc="Extracting MFCCs"):
        wav = load_audio(path)
        feats.append(extract_mfcc_vector(wav))
    return np.stack(feats)


def compute_eer(y_true, y_score):
    """Equal Error Rate — the standard ASVspoof metric, used so your results
    are directly comparable to numbers reported in the literature."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def train_and_eval(train_df, eval_df, model_type="logreg"):
    X_train = featurize_manifest(train_df)
    y_train = train_df.label_binary.values
    X_eval = featurize_manifest(eval_df)
    y_eval = eval_df.label_binary.values

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": model_type,
        "accuracy": accuracy_score(y_eval, y_pred),
        "f1": f1_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred),
        "recall": recall_score(y_eval, y_pred),
        "eer": compute_eer(y_eval, y_prob),
    }
    return clf, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model_type", default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--out_dir", default="checkpoints/baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)

    clf, metrics = train_and_eval(train_df, eval_df, args.model_type)

    joblib.dump(clf, Path(args.out_dir) / f"{args.model_type}.joblib")
    with open(Path(args.out_dir) / f"{args.model_type}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
