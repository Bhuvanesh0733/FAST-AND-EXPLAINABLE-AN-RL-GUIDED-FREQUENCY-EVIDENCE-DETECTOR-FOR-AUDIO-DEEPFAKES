"""
NOT part of the deliverable — this only generates fake random-noise .flac
files + a matching protocol.txt so we can smoke-test the pipeline code
(data_prep -> features -> baseline -> RL stage) without the real 7GB
ASVspoof2019 LA download. Delete this whole tests/ folder before submitting;
it's scaffolding, not part of the project.
"""
import os
import numpy as np
import soundfile as sf

SR = 16000
ROOT = "tests/fake_LA"


def make_clip(path, seconds=2.0, seed=0, band_bias=None):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    wav = rng.normal(0, 0.05, n).astype(np.float32)
    if band_bias is not None:
        # add a sine burst to simulate a "band artifact" for fake clips
        t = np.arange(n) / SR
        wav += 0.05 * np.sin(2 * np.pi * band_bias * t)
    sf.write(path, wav, SR)


def main():
    for split, n_speakers in [("train", 2), ("dev", 1), ("eval", 1)]:
        flac_dir = f"{ROOT}/ASVspoof2019_LA_{split}/flac"
        os.makedirs(flac_dir, exist_ok=True)

    protocol_dir = f"{ROOT}/ASVspoof2019_LA_cm_protocols"
    os.makedirs(protocol_dir, exist_ok=True)

    protocols = {
        "train": ("ASVspoof2019.LA.cm.train.trn.txt", 40, ["A01", "A02"]),
        "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", 20, ["A01", "A02"]),
        "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", 20, ["A07", "A08"]),  # "unseen"
    }

    seed = 0
    for split, (fname, n_clips, attack_ids) in protocols.items():
        lines = []
        flac_dir = f"{ROOT}/ASVspoof2019_LA_{split}/flac"
        for i in range(n_clips):
            seed += 1
            is_spoof = i % 2 == 0
            file_name = f"LA_{split.upper()[0]}_{1000+i}"
            speaker = f"LA_{seed % 3:04d}"
            if is_spoof:
                attack = attack_ids[i % len(attack_ids)]
                label = "spoof"
                make_clip(f"{flac_dir}/{file_name}.flac", seed=seed, band_bias=3000)
            else:
                attack = "-"
                label = "bonafide"
                make_clip(f"{flac_dir}/{file_name}.flac", seed=seed)
            lines.append(f"{speaker} {file_name} - {attack} {label}")
        with open(f"{protocol_dir}/{fname}", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{split}: wrote {n_clips} synthetic clips")


if __name__ == "__main__":
    main()
