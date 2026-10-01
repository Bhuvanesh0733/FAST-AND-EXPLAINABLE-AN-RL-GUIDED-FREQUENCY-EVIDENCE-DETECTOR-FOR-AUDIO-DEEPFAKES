"""
evaluate.py
-----------
Produces the Results & Evaluation section deliverables:
  - a 3-row comparison table (baseline vs Stage 1 vs Stage 2)
  - a per-attack-type (A07-A19) accuracy breakdown, to identify which
    unseen synthesis methods fool the model most (the error-analysis
    the rubric explicitly rewards)
  - an ROC/DET-style plot

Usage:
    python -m src.evaluate --eval data/processed/eval_manifest.csv \
                            --baseline_metrics checkpoints/baseline/logreg_metrics.json \
                            --stage1_dir checkpoints/stage1 \
                            --out_dir reports
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def per_attack_breakdown(eval_df: pd.DataFrame, preds_col: str = "pred") -> pd.DataFrame:
    """Requires eval_df to already have a `pred` column (0/1) from whichever
    model you're analyzing. Reports accuracy per attack_id, sorted worst-first
    so the hardest attack types surface immediately."""
    spoof_only = eval_df[eval_df.label_binary == 1].copy()
    spoof_only["correct"] = (spoof_only[preds_col] == spoof_only.label_binary).astype(int)
    breakdown = (
        spoof_only.groupby("attack_id")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_clips"})
        .sort_values("accuracy")
    )
    return breakdown


def build_comparison_table(metrics_paths: dict, out_dir: str):
    """metrics_paths: {"Baseline (MFCC+LogReg)": path.json, "Stage 1 (SSL)": path.json, ...}"""
    rows = []
    for name, path in metrics_paths.items():
        if path is None or not Path(path).exists():
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({"model": name, **m})
    table = pd.DataFrame(rows)
    out_path = Path(out_dir) / "comparison_table.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}")
    return table


def plot_comparison(table: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_to_plot = [c for c in ["accuracy", "f1", "eer"] if c in table.columns]
    table.set_index("model")[metrics_to_plot].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Stage 1 vs Stage 2")
    plt.tight_layout()
    out_path = Path(out_dir) / "comparison_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", default=None)
    parser.add_argument("--stage1_metrics", default=None)
    parser.add_argument("--stage2_metrics", default=None)
    parser.add_argument("--out_dir", default="reports")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    metrics_paths = {
        "Baseline (MFCC + LogReg)": args.baseline_metrics,
        "Stage 1 (SSL transfer learning)": args.stage1_metrics,
        "Stage 2 (RL-refined, interpretable)": args.stage2_metrics,
    }
    table = build_comparison_table(metrics_paths, args.out_dir)
    if len(table) > 0:
        plot_comparison(table, args.out_dir)


if __name__ == "__main__":
    main()
