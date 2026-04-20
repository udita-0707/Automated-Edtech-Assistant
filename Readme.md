# Automated EdTech Grading Assistant

**Domain:** Document AI + NLP + Education Technology  
**Objective:** A two-phase pipeline for automatically transcribing handwritten student answers and grading them against reference solutions using semantic similarity.

---

## 🏗️ System Architecture

The following diagram illustrates the end-to-end flow from a handwritten image to a final grade and feedback.

```text
+-------------------+      +-----------------------+      +-----------------------+
|  Student Image    | ---> |   OCR Engine          | ---> |   Text Cleaning       |
|  (Handwriting)    |      | (Phase 1: Tesseract)  |      | (Preprocessing &      |
|                   |      | (Phase 2: TrOCR)      |      |  Normalization)       |
+-------------------+      +-----------------------+      +-----------------------+
                                                                     |
                                                                     v
+-------------------+      +-----------------------+      +-----------------------+
|   Final Output    | <--- |   Grading Model       | <--- |   Feature Extraction  |
|  (Grade + Feedback)      | (Phase 1: Cosine/LogR)|      | (TF-IDF / SBERT       |
|                   |      | (Phase 2: Hybrid SVM) |      |  Embeddings)          |
+-------------------+      +-----------------------+      +-----------------------+
```

---

## 🚀 Reproduce Locally

### 1. Environment Requirements
* **Python:** 3.11 or 3.12 (Recommended). Avoid 3.14 alpha/beta due to library compatibility issues.
* **System Tools:** `tesseract` (for Phase 1 OCR).

### 2. Setup Instructions
```bash
# Clone the repository
git clone <repository-url>
cd Automated-Edtech-Assistant

# Create and activate a Virtual Environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r ml-service/requirements.txt
# Additional Phase 2 requirements
pip install torch transformers sentence-transformers
```

### 3. macOS Troubleshooting
If you encounter the `externally-managed-environment` error:
> This occurs on macOS when trying to install packages system-wide. Always ensure your virtual environment is activated (`source venv/bin/activate`) before running pip.

### 4. 🛠️ Generating Model Artifacts (Crucial)
Following industry best practices, large binary artifacts (`.pkl` files) are not tracked in this repository to ensure reproducibility and security. You **must** generate these files locally before running the API or frontend:

```bash
# Generate Phase 1 Model (Logistic Regression + Vectorizer)
python ml-service/run_train_eval.py

# Generate Phase 2 Model (SVM Ensemble + SBERT artifacts)
python phase2/run_train_eval.py
```
This process validates that the code is fully operational on your local machine.

---

## 🔬 Phase 1: Classical ML Pipeline

**Approach:** TF-IDF Vectorization + Cosine Similarity with a Logistic Regression wrapper.
* **Why TF-IDF?** It is highly efficient for keyword-matching and capturing structural overlap in short scientific answers.

### Execution
```bash
cd ml-service
# Train the baseline model on SciEntsBank dataset
python run_train_eval.py
```

### Samples Output (JSON)
```json
{
  "predicted_label": "correct",
  "similarity_score": 0.82,
  "confidence": 0.94,
  "feedback": "Great work! Your answer matches the reference closely."
}
```

---

## 🧠 Phase 2: Deep Learning Pipeline

**Approach:** Transformer-based Hybrid Pipeline.
* **OCR:** TrOCR (ViT-Encoder + RoBERTa-Decoder) for robust handwriting recognition.
* **Grading:** Hybrid Ensemble (Model C) combining SVM (Keyword precision) + SBERT (Semantic depth).
* **Why SBERT?** It uses Siamese networks to understand the *meaning* of sentences even when keywords differ (e.g., "powerhouse" vs "generates ATP").

### Execution
```bash
# Run the Phase 2 ablation study
python phase2/run_train_eval.py
```

---

## 📂 Project Structure

```text
Automated-Edtech-Assistant/
├── ml-service/           # Live ML Backend (FastAPI)
├── backend/              # Node.js/Express API with SQLite
├── frontend/             # React UI with OCR scanning
├── phase1/               # Phase 1 submission artifacts
├── phase2/               # Phase 2 submission artifacts
├── notebooks/            # Research & prototyping notebooks
└── requirements.txt      # Project dependencies
```

---

## ⚠️ Common Issues

1. **`ModuleNotFoundError: No module named 'pandas'`**
   * **Fix:** Ensure you are running Python from within the `venv`. Run `which python` to verify.
2. **`pydantic-core` build failure**
   * **Fix:** Your Python version is too new (3.14). Downgrade to Python 3.11.
3. **Noisy OCR Output**
   * **Fix:** Phase 1 requires high-contrast images. Use the `TesseractOCR.preprocess()` method to binarize images before transcription.
4. **Low Similarity Scores**
   * **Fix:** Scientific terms can be specific. Ensure your reference answer contains the technical keywords expected by the model.

---

## 🎓 Evaluator Metrics
* **Reproducibility:** Confirmed on Python 3.11 (macOS aarch64).
* **Architecture:** Modular separation between OCR, Preprocessing, and Grading.
* **Engineering:** Includes SQLite persistence for history and weighted ensemble grading logic.
