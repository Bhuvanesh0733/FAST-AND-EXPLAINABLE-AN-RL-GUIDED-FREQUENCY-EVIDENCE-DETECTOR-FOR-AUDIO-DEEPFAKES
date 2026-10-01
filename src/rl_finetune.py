"""
rl_finetune.py
--------------
Stage 2: the RL-inspired interpretability layer. This is the part of the
project that directly answers the case-study gap (AASIST = fast+opaque,
FT-GRPO = interpretable+heavy): a small policy network learns WHICH
frequency bands of the spectrogram are the most useful evidence for the
real/fake decision, trained with a GRPO-style group-relative policy
gradient — the same reward-shaping idea as FT-GRPO's GRPO stage, just
applied to frequency-band selection instead of natural-language reasoning,
which makes it small enough to train on CPU in minutes.

Two models here:
  1. EvidenceClassifier — a small supervised MLP on band-energy features.
     This is the "task" whose decisions we're about to explain. It is
     frozen once trained; the RL stage never updates it.
  2. EvidencePolicy — the RL-trained component. For each clip, it outputs
     a keep/drop probability per frequency band. We sample a *group* of
     G candidate band-subsets per clip (like GRPO's group sampling),
     score each subset by whether the frozen classifier still gets the
     right answer using ONLY those bands (plus a sparsity penalty so it
     can't just keep everything), and update the policy using the
     group's mean reward as the baseline (group-relative advantage).

This stage needs no internet access and no GPU — it runs entirely on the
band-energy features from features.py.

Usage:
    python -m src.rl_finetune train_classifier --train data/processed/train_manifest.csv \
                                                --dev data/processed/dev_manifest.csv \
                                                --out_dir checkpoints/stage2
    python -m src.rl_finetune train_policy --train data/processed/train_manifest.csv \
                                            --dev data/processed/dev_manifest.csv \
                                            --out_dir checkpoints/stage2
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli
from tqdm import tqdm

from .features import N_BANDS, extract_band_energy_features, load_audio

FEATURE_DIM = 2 * N_BANDS  # mean + std per band


class EvidenceClassifier(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)


class EvidencePolicy(nn.Module):
    """Outputs a per-band keep probability given the full band-feature vector."""

    def __init__(self, in_dim=FEATURE_DIM, n_bands=N_BANDS, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_bands)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.sigmoid(logits)  # per-band keep probability


def featurize(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for path in tqdm(df.file_path, desc="Extracting band-energy features"):
        wav = load_audio(path)
        feats.append(extract_band_energy_features(wav))
    return np.stack(feats)


def apply_band_mask(x: torch.Tensor, mask: torch.Tensor, n_bands: int) -> torch.Tensor:
    """x: (B, 2*n_bands) = [means..., stds...]; mask: (B, n_bands) binary.
    Zeroes out both the mean and std of any dropped band."""
    full_mask = torch.cat([mask, mask], dim=1)  # apply same mask to means and stds
    return x * full_mask


def train_classifier(args):
    train_df = pd.read_csv(args.train)
    dev_df = pd.read_csv(args.dev)
    X_train = torch.tensor(featurize(train_df), dtype=torch.float32)
    y_train = torch.tensor(train_df.label_binary.values, dtype=torch.long)
    X_dev = torch.tensor(featurize(dev_df), dtype=torch.float32)
    y_dev = torch.tensor(dev_df.label_binary.values, dtype=torch.long)

    model = EvidenceClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_preds = model(X_dev).argmax(dim=1)
            dev_acc = accuracy_score(y_dev.numpy(), dev_preds.numpy())
        print(f"Epoch {epoch+1}: train_loss={loss.item():.4f} dev_acc={dev_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), Path(args.out_dir) / "evidence_classifier.pt")
    return model


def train_policy(args, classifier: EvidenceClassifier = None):
    """GRPO-style training: for each clip, sample a GROUP of G band-subsets,
    reward each by (classifier still correct using only those bands) minus
    a sparsity penalty, then update the policy using the group's own mean
    reward as the baseline — no separate value network needed, same trick
    FT-GRPO uses to avoid training a critic."""
    train_df = pd.read_csv(args.train)
    dev_df = pd.read_csv(args.dev)
    X_train = torch.tensor(featurize(train_df), dtype=torch.float32)
    y_train = torch.tensor(train_df.label_binary.values, dtype=torch.long)

    if classifier is None:
        classifier = EvidenceClassifier()
        classifier.load_state_dict(torch.load(Path(args.out_dir) / "evidence_classifier.pt"))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    policy = EvidencePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    G = args.group_size  # candidates sampled per clip, like GRPO's group size
    sparsity_weight = args.sparsity_weight

    for epoch in range(args.epochs):
        policy.train()
        perm = torch.randperm(len(X_train))
        total_reward = 0.0
        n_steps = 0

        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            x = X_train[idx]  # (B, F)
            y = y_train[idx]  # (B,)
            probs = policy(x)  # (B, n_bands)

            # sample a GROUP of G mask candidates per example
            group_log_probs, group_rewards = [], []
            for g in range(G):
                dist = Bernoulli(probs)
                mask = dist.sample()  # (B, n_bands)
                log_prob = dist.log_prob(mask).sum(dim=1)  # (B,)

                masked_x = apply_band_mask(x, mask, N_BANDS)
                with torch.no_grad():
                    pred = classifier(masked_x).argmax(dim=1)
                r_acc = torch.where(pred == y, 1.0, -1.0)
                r_sparsity = -sparsity_weight * mask.sum(dim=1)  # fewer bands kept = smaller penalty
                reward = r_acc + r_sparsity

                group_log_probs.append(log_prob)
                group_rewards.append(reward)

            group_log_probs = torch.stack(group_log_probs)  # (G, B)
            group_rewards = torch.stack(group_rewards)  # (G, B)
            baseline = group_rewards.mean(dim=0, keepdim=True)  # group-relative baseline
            advantage = group_rewards - baseline

            loss = -(group_log_probs * advantage.detach()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += group_rewards.mean().item()
            n_steps += 1

        print(f"Epoch {epoch+1}: mean_group_reward={total_reward/n_steps:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(policy.state_dict(), Path(args.out_dir) / "evidence_policy.pt")
    return policy


def explain(wav: np.ndarray, classifier: EvidenceClassifier, policy: EvidencePolicy):
    """Inference helper used by the app: returns prediction, confidence,
    and which frequency bands were kept as evidence."""
    feats = torch.tensor(extract_band_energy_features(wav), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = policy(feats)
        keep_mask = (probs >= 0.5).float()
        masked_feats = apply_band_mask(feats, keep_mask, N_BANDS)
        logits = classifier(masked_feats)
        pred_probs = torch.softmax(logits, dim=1)[0]
    return {
        "prediction": "fake" if pred_probs[1] > pred_probs[0] else "real",
        "confidence": float(pred_probs.max()),
        "evidence_bands": keep_mask[0].numpy().tolist(),  # which of the N_BANDS were used
        "band_keep_probs": probs[0].numpy().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("train_classifier")
    p1.add_argument("--train", required=True)
    p1.add_argument("--dev", required=True)
    p1.add_argument("--out_dir", default="checkpoints/stage2")
    p1.add_argument("--epochs", type=int, default=200)

    p2 = sub.add_parser("train_policy")
    p2.add_argument("--train", required=True)
    p2.add_argument("--dev", required=True)
    p2.add_argument("--out_dir", default="checkpoints/stage2")
    p2.add_argument("--epochs", type=int, default=20)
    p2.add_argument("--batch_size", type=int, default=32)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--group_size", type=int, default=4, help="G in GRPO-style group sampling")
    p2.add_argument("--sparsity_weight", type=float, default=0.05)

    args = parser.parse_args()
    if args.command == "train_classifier":
        train_classifier(args)
    elif args.command == "train_policy":
        train_policy(args)


if __name__ == "__main__":
    main()
