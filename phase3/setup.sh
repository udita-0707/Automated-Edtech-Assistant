#!/bin/bash
# Phase 3 Reproducibility Script
# ==========================================
# This script sets up the Python environment, installs all dependencies
# across all phases, downloads required NLTK/spaCy data, and executes
# the data pipeline to generate the Phase 3 artifacts.
#
# Usage: bash phase3/setup.sh

set -e # Exit immediately if a command exits with a non-zero status

# Ensure we are in the root directory
if [ ! -d "phase3" ]; then
    echo "❌ Error: Please run this script from the project root."
    echo "Usage: bash phase3/setup.sh"
    exit 1
fi

echo "=========================================="
echo "Phase 3 Reproducibility Setup"
echo "=========================================="

echo "[1/4] Creating virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created."
else
    echo "✓ Virtual environment already exists."
fi

# Activate the virtual environment
source .venv/bin/activate

echo "[2/4] Installing dependencies..."
# Upgrade pip first
pip install --upgrade pip > /dev/null

# Install phase 3 requirements (which is a superset of phase 1 and 2)
if [ -f "phase3/requirements.txt" ]; then
    pip install -r phase3/requirements.txt
    echo "✓ Dependencies installed."
else
    echo "❌ Error: phase3/requirements.txt not found."
    exit 1
fi

echo "[3/4] Downloading NLP datasets (spaCy & NLTK)..."
# spaCy en_core_web_sm model
python -m spacy download en_core_web_sm
# NLTK punkt
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True);"
echo "✓ NLP models downloaded."

echo "[4/4] Executing Phase 3 Data Pipeline..."
echo "This will calibrate SBERT thresholds, run the ablation study, and generate the bias report."
echo "------------------------------------------"
python phase3/run_pipeline.py
echo "------------------------------------------"

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo "You can now start the API with:"
echo "    source .venv/bin/activate"
echo "    cd phase3/api && uvicorn main:app --port 8002"
