# EdTech ML Service - Answer Grader

## Architecture & Code Quality
This service hosts the `LogisticRegression` classifier that evaluates textual semantic similarity.

The repository follows standard software engineering practices:
- **Modular Structure**: Logic is cleanly separated into routing (`main.py`), ML classes (`model.py`), and heuristic helpers (`utils.py`).
- **Idempotency**: The model caches the trained weights in `data/model.pkl` to prevent redundant downloads.
- **Robust Typing**: Type hints and Pydantic validation are utilized across all components.

## Running the Service
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the FastAPI development server:
   ```bash
   uvicorn main:app --reload
   ```

## Design Decisions
- We chose **Logistic Regression** over Deep Learning or SVMs because it offers absolute explainability for the exact influence of lengths vs semantic overlaps, satisfying theoretical clarity.
- We utilize TF-IDF to create numerical embedding arrays of the `nkazi/SciEntsBank` ground-truth corpus. Length ratios have been statistically bounded to minimize explosive outliers.
