#!/bin/bash
set -e
apt-get install -y ffmpeg
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
