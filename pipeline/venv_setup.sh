#!/usr/bin/env bash
set -e

echo "Setting up Python virtual environment..."
python3.11 -m venv pipeline/.venv

source pipeline/.venv/Scripts/activate || source pipeline/.venv/bin/activate

pip install --upgrade pip

echo "Installing CPU-only PyTorch..."
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchaudio==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu

echo "Installing requirements.txt..."
pip install -r pipeline/requirements.txt

echo "Verifying pipeline dependencies..."
python -c "import faster_whisper; import torch; print('Pipeline deps OK')"
