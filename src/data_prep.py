"""
data_prep.py
------------
Parses the ASVspoof2019 LA protocol files into clean pandas manifests
(train / dev / eval) with columns: speaker_id, file_name, attack_id, label, split.

ASVspoof2019 LA protocol line format (standard, unchanged since 2019):
    SPEAKER_ID  AUDIO_FILE_NAME  -  ATTACK_ID  KEY
    e.g. "LA_0079 LA_T_1138215 - - bonafide"
    e.g. "LA_0079 LA_T_1271820 - A01 spoof"

If your downloaded README.txt shows a different column order, adjust
PROTOCOL_COLUMNS below to match — everything downstream reads from the
named columns, not positions, so one edit here is all it takes.

Usage:
    python -m src.data_prep --la_root /path/to/LA --out_dir data/processed
"""
import argparse
import os
from pathlib import Path

import pandas as pd

PROTOCOL_COLUMNS = ["speaker_id", "file_name", "env_id", "attack_id", "label"]

# Attack IDs seen during train/dev — anything else in eval is an "unseen attack"
# (this is what makes ASVspoof2019 LA a genuine generalization test, not just
# a train/test split of the same distribution)
KNOWN_TRAIN_ATTACKS = {"A01", "A02", "A03", "A04", "A05", "A06"}


def parse_protocol(protocol_path: str, split: str) -> pd.DataFrame:
    """Parse one ASVspoof2019 LA protocol .txt file into a DataFrame."""
    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != len(PROTOCOL_COLUMNS):
                raise ValueError(
                    f"Unexpected column count in {protocol_path}: "
                    f"got {len(parts)} fields, expected {len(PROTOCOL_COLUMNS)}. "
                    f"Check README.txt for the current protocol format and "
                    f"update PROTOCOL_COLUMNS in data_prep.py."
                )
            rows.append(parts)
    df = pd.DataFrame(rows, columns=PROTOCOL_COLUMNS)
    df["split"] = split
    df["label_binary"] = (df["label"] == "spoof").astype(int)  # 1 = fake, 0 = real
    df["is_unseen_attack"] = ~df["attack_id"].isin(KNOWN_TRAIN_ATTACKS) & (
        df["label"] == "spoof"
    )
    return df


def attach_file_paths(df: pd.DataFrame, la_root: str, split: str) -> pd.DataFrame:
    """Add absolute .flac file paths. ASVspoof2019 LA layout:
    LA/ASVspoof2019_LA_{split}/flac/{file_name}.flac
    """
    split_dir = {"train": "train", "dev": "dev", "eval": "eval"}[split]
    flac_dir = Path(la_root) / f"ASVspoof2019_LA_{split_dir}" / "flac"
    df = df.copy()
    df["file_path"] = df["file_name"].apply(lambda x: str(flac_dir / f"{x}.flac"))
    return df


def build_manifests(la_root: str, protocol_dir: str) -> dict:
    """Build train/dev/eval manifests. protocol_dir is the folder containing
    the ASVspoof2019.LA.cm.*.txt protocol files (usually LA/ASVspoof2019_LA_cm_protocols/).
    """
    files = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    manifests = {}
    for split, fname in files.items():
        protocol_path = Path(protocol_dir) / fname
        df = parse_protocol(str(protocol_path), split)
        df = attach_file_paths(df, la_root, split)
        manifests[split] = df
    return manifests


def stratified_subsample(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """Subsample a manifest to n_per_class real + n_per_class fake, keeping
    attack-type proportions roughly intact. Use this ONLY on train/dev —
    never subsample the eval set, or you lose the unseen-attack generalization test.
    """
    real = df[df.label_binary == 0]
    fake = df[df.label_binary == 1]
    real_s = real.sample(n=min(n_per_class, len(real)), random_state=seed)
    fake_s = fake.sample(n=min(n_per_class, len(fake)), random_state=seed)
    return pd.concat([real_s, fake_s]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--la_root", required=True, help="Path to unzipped LA/ folder")
    parser.add_argument(
        "--protocol_dir",
        default=None,
        help="Path to ASVspoof2019_LA_cm_protocols/ (defaults to <la_root>/ASVspoof2019_LA_cm_protocols)",
    )
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument(
        "--subsample_train",
        type=int,
        default=0,
        help="If >0, subsample train to this many clips per class (compute-feasible mode)",
    )
    parser.add_argument(
        "--subsample_dev",
        type=int,
        default=0,
        help="If >0, subsample dev to this many clips per class",
    )
    args = parser.parse_args()

    protocol_dir = args.protocol_dir or str(Path(args.la_root) / "ASVspoof2019_LA_cm_protocols")
    os.makedirs(args.out_dir, exist_ok=True)

    manifests = build_manifests(args.la_root, protocol_dir)

    if args.subsample_train > 0:
        manifests["train"] = stratified_subsample(manifests["train"], args.subsample_train)
    if args.subsample_dev > 0:
        manifests["dev"] = stratified_subsample(manifests["dev"], args.subsample_dev)
    # eval is NEVER subsampled — see docstring above

    for split, df in manifests.items():
        out_path = Path(args.out_dir) / f"{split}_manifest.csv"
        df.to_csv(out_path, index=False)
        n_real = (df.label_binary == 0).sum()
        n_fake = (df.label_binary == 1).sum()
        print(f"[{split}] {len(df)} clips ({n_real} real / {n_fake} fake) -> {out_path}")
        if split == "eval":
            n_unseen = df.is_unseen_attack.sum()
            print(f"  -> {n_unseen} clips use attack types unseen in training (A07-A19)")


if __name__ == "__main__":
    main()
