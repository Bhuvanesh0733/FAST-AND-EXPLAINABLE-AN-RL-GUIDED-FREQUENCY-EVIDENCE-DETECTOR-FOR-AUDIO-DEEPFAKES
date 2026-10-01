"""
ssl_classifier.py
------------------
Stage 1: the main "advanced ML model" for the assignment — transfer learning
from a pretrained self-supervised speech encoder (Wav2Vec2 or WavLM) plus a
lightweight attention-pooling classification head.

This corresponds to the "fast, deployable" half of the case-study gap (the
AASIST side): it's a single small head on top of a frozen/lightly fine-tuned
encoder, not a multi-billion-parameter LLM.

Requires internet access to download pretrained weights from huggingface.co
the first time you run it (works fine on Colab; will NOT work in a fully
offline/sandboxed environment).

Usage:
    python -m src.ssl_classifier train --train data/processed/train_manifest.csv \
                                        --dev data/processed/dev_manifest.csv \
                                        --out_dir checkpoints/stage1 \
                                        --epochs 10
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model

from .features import AudioDataset

ENCODER_NAME = "facebook/wav2vec2-base"  # swap for "microsoft/wavlm-base" if preferred


class AttentionPool(nn.Module):
    """Learns which timesteps of the SSL encoder's output matter most,
    instead of naive mean-pooling."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B, T, H)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)  # (B, H)
        return pooled, weights.squeeze(-1)  # also return weights for inspection/app overlay


class SSLClassifier(nn.Module):
    def __init__(self, encoder_name: str = ENCODER_NAME, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(encoder_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden_dim = self.encoder.config.hidden_size
        self.pool = AttentionPool(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, waveform):  # waveform: (B, T_samples)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                features = self.encoder(waveform).last_hidden_state
        else:
            features = self.encoder(waveform).last_hidden_state
        pooled, attn_weights = self.pool(features)
        logits = self.classifier(pooled)
        return logits, attn_weights


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            wav = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(wav)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds),
        "eer": compute_eer(all_labels, all_probs),
    }


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    import pandas as pd

    train_df = pd.read_csv(args.train)
    dev_df = pd.read_csv(args.dev)
    train_loader = DataLoader(AudioDataset(train_df), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(AudioDataset(dev_df), batch_size=args.batch_size)

    model = SSLClassifier(freeze_encoder=args.freeze_encoder).to(device)
    # class-weighted loss: real (bonafide) clips are far rarer than fake ones
    n_real = (train_df.label_binary == 0).sum()
    n_fake = (train_df.label_binary == 1).sum()
    class_weights = torch.tensor(
        [len(train_df) / (2 * n_real), len(train_df) / (2 * n_fake)], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_eer = float("inf")
    patience, bad_epochs = 3, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            wav = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits, _ = model(wav)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        dev_metrics = evaluate(model, dev_loader, device)
        print(
            f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} dev_eer={dev_metrics['eer']:.4f}"
        )

        if dev_metrics["eer"] < best_eer:
            best_eer = dev_metrics["eer"]
            bad_epochs = 0
            torch.save(model.state_dict(), Path(args.out_dir) / "best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    with open(Path(args.out_dir) / "train_summary.json", "w") as f:
        json.dump({"best_dev_eer": best_eer}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--dev", required=True)
    p_train.add_argument("--out_dir", default="checkpoints/stage1")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--freeze_encoder", action="store_true", default=True)

    args = parser.parse_args()
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
