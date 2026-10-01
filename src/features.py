"""
features.py
------------
Audio loading + feature extraction shared by the baseline model, the Stage-1
SSL classifier, and the Stage-2 evidence/explanation module.

Two feature views of the same clip are used on purpose:
  - MFCCs (handcrafted)      -> baseline model, and the rule-based evidence
                                 reward in Stage 2 (needs an interpretable,
                                 non-learned signal to check against)
  - Raw waveform             -> fed straight into the pretrained SSL encoder
                                 (Wav2Vec2 / WavLM) for the Stage-1/2 models
"""
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
CLIP_SECONDS = 2.0
CLIP_LEN = int(SAMPLE_RATE * CLIP_SECONDS)


def load_audio(path: str, sr: int = SAMPLE_RATE, clip_len: int = CLIP_LEN) -> np.ndarray:
    """Load an audio file, resample to `sr`, and pad/trim to a fixed length
    so clips can be batched. Short clips are looped (common ASVspoof practice)
    rather than zero-padded, so the model always sees real signal.
    """
    wav, _ = librosa.load(path, sr=sr, mono=True)
    if len(wav) == 0:
        wav = np.zeros(clip_len, dtype=np.float32)
    if len(wav) < clip_len:
        n_repeats = int(np.ceil(clip_len / len(wav)))
        wav = np.tile(wav, n_repeats)
    wav = wav[:clip_len]
    # peak normalize
    peak = np.max(np.abs(wav)) + 1e-9
    wav = wav / peak
    return wav.astype(np.float32)


def extract_mfcc_vector(wav: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = 20) -> np.ndarray:
    """Mean+std pooled MFCCs -> a fixed-length vector for the classical baseline
    (Logistic Regression / Random Forest can't consume variable-length sequences).
    """
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extract_melspectrogram(wav: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 64) -> np.ndarray:
    """Log-mel spectrogram (time x freq) — used for visualization in the app
    and as the region grid the Stage-2 evidence head attends over.
    """
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


N_BANDS = 8  # number of frequency bands used by the Stage-2 evidence/RL module


def extract_band_energy_features(
    wav: np.ndarray, sr: int = SAMPLE_RATE, n_bands: int = N_BANDS, n_mels: int = 64
) -> np.ndarray:
    """Splits the mel-spectrogram into `n_bands` contiguous frequency bands and
    returns [mean_1..mean_n, std_1..std_n] log-energy per band.
    This is the compact, fully-interpretable feature set the Stage-2 RL policy
    learns to select from (i.e. "which frequency bands look suspicious") —
    deliberately simple and non-learned so masking a band has a clear,
    human-checkable meaning, unlike masking an opaque SSL embedding dimension.
    """
    log_mel = extract_melspectrogram(wav, sr=sr, n_mels=n_mels)
    band_edges = np.linspace(0, n_mels, n_bands + 1).astype(int)
    means, stds = [], []
    for i in range(n_bands):
        band = log_mel[band_edges[i] : band_edges[i + 1], :]
        means.append(band.mean())
        stds.append(band.std())
    return np.array(means + stds, dtype=np.float32)  # length 2*n_bands


class AudioDataset(Dataset):
    """PyTorch Dataset over a manifest DataFrame (see data_prep.py).
    Returns raw waveform tensors — the SSL encoder in ssl_classifier.py
    does its own internal feature extraction from raw audio.
    """

    def __init__(self, manifest_df, sr: int = SAMPLE_RATE, clip_len: int = CLIP_LEN):
        self.df = manifest_df.reset_index(drop=True)
        self.sr = sr
        self.clip_len = clip_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = load_audio(row.file_path, sr=self.sr, clip_len=self.clip_len)
        label = int(row.label_binary)
        return {
            "waveform": torch.tensor(wav, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "attack_id": row.attack_id,
            "file_name": row.file_name,
        }
