# Configuration settings for the Automated EdTech Assistant
# Centralized hyperparameters and file paths

DATASET_NAME = "nkazi/SciEntsBank"

# Label scheme options: "5way", "3way", "2way"
# Mapping provided by user: correct->0, contradictory->1, others->2
LABEL_SCHEME = "3way"

# Maximum number of features for TF-IDF
MAX_FEATURES = 5000

# Random state for reproducibility
RANDOM_STATE = 42

# Paths
MODEL_PATH = "data/model.pkl"
SCALER_PATH = "data/scaler.pkl"
VECTORIZER_PATH = "data/vectorizer.pkl"
