# Configuration for Phase 2 Neural Pipeline

import os

DATASET_NAME = "nkazi/SciEntsBank"
LABEL_SCHEME = "3way"

# Model A (SVM) Paths
MODEL_A_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "artifacts", "model_a.pkl")

# Model B/C (SBERT/Hybrid) Hyperparameters
HYBRID_ALPHA = 0.4
SIMILARITY_THRESHOLD_CORRECT = 0.75
SIMILARITY_THRESHOLD_PARTIAL = 0.45

# OCR Model
TROCR_MODEL = "microsoft/trocr-base-handwritten"

RANDOM_STATE = 42
