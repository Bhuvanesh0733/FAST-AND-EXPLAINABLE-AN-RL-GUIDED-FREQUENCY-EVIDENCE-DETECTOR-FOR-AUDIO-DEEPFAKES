# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies (ffmpeg for MP3 support) ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install CPU-only PyTorch first (avoids pulling 2GB CUDA build) ────────────
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining dependencies ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY app/        ./app/
COPY src/        ./src/
COPY checkpoints/ ./checkpoints/

# ── Pre-download Wav2Vec2 weights so it works offline at runtime ──────────────
RUN python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')"

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Run with gunicorn ─────────────────────────────────────────────────────────
CMD ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:5000", "app.server:app"]
