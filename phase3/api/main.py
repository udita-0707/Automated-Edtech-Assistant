"""
Phase 3 — FastAPI Service (Port 8002)
========================================
Serves the SynergisticHybridGrader with concept-level feedback.

Node backend contract (unchanged from Phase 1/2):
    POST /predict
    Receives: {question, student_answer, reference_answer}
    Returns:  {predicted_label, similarity_score, confidence, feedback}
    + Phase 3 extensions: {path, stage, concepts_missing,
                           concepts_present, completeness_score, topic_gap}

All Phase 1/2 field names are preserved so the Node backend
(backend/index.js) requires zero changes.

Extra endpoints:
    GET  /explain?student_answer=...&reference_answer=...
    POST /ocr     — multipart image → TrOCR transcription
    GET  /health  — model status
"""

import io
import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Resolve repo root
_HERE  = os.path.dirname(os.path.abspath(__file__))   # phase3/api
_PH3   = os.path.dirname(_HERE)                        # phase3
_ROOT  = os.path.dirname(_PH3)                         # repo root
_PH2   = os.path.join(_ROOT, "phase2")
# Priority: phase3 > root > phase2  (last insert wins position 0)
for _p in [_ROOT, _PH2, _PH3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
    else:
        sys.path.remove(_p)
        sys.path.insert(0, _p)
if sys.path[0] != _PH3:
    sys.path.insert(0, _PH3)

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry (populated at startup)
# ---------------------------------------------------------------------------
_models = {}


# ---------------------------------------------------------------------------
# Lifespan startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models once at startup.

    Loading order:
    1. ClassicalGrader (SVM) — from phase2/data/artifacts/model_a.pkl
    2. CalibratedScorer (SBERT) — load thresholds from JSON or calibrate
    3. SynergisticHybridGrader — couple the two above
    4. FeedbackGenerator (spaCy)
    5. HandwritingOCR (TrOCR) — lazy, only if requested

    All errors are caught with descriptive messages so the service
    can start even if optional dependencies (spaCy, TrOCR) are missing.
    """
    logger.info("=" * 50)
    logger.info("Phase 3 API — Model Loading")
    logger.info("=" * 50)

    # -- ClassicalGrader (Phase 2 SVM) ------------------------------------
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "classical_grader",
            os.path.join(_PH2, "grading", "classical_grader.py")
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        ClassicalGrader = _mod.ClassicalGrader
        cg = ClassicalGrader()
        if not os.path.exists(config.MODEL_A_PATH):
            raise FileNotFoundError(
                f"Phase 2 SVM artefact not found at {config.MODEL_A_PATH}. "
                "Run: python phase2/run_train_eval.py"
            )
        cg.load(config.MODEL_A_PATH)
        _models["classical"] = cg
        logger.info(f"✓ ClassicalGrader loaded from {config.MODEL_A_PATH}")
    except Exception as exc:
        logger.error(f"✗ ClassicalGrader failed to load: {exc}")
        _models["classical"] = None

    # -- CalibratedScorer (SBERT) ----------------------------------------
    try:
        from grading.calibrated_scorer import CalibratedScorer  # noqa
        cs = CalibratedScorer()
        if not cs.load_thresholds():
            logger.warning(
                "Calibrated thresholds not found. "
                "Run python phase3/run_pipeline.py to calibrate first."
            )
        _models["calibrated"] = cs
        logger.info(f"✓ CalibratedScorer loaded (thresholds: {cs.thresholds})")
    except Exception as exc:
        logger.error(f"✗ CalibratedScorer failed: {exc}")
        _models["calibrated"] = None

    # -- SynergisticHybridGrader ------------------------------------------
    if _models.get("classical") and _models.get("calibrated"):
        try:
            from grading.hybrid_grader import SynergisticHybridGrader  # noqa
            hg = SynergisticHybridGrader(
                classical_grader=_models["classical"],
                calibrated_scorer=_models["calibrated"],
            )
            _models["hybrid"] = hg
            logger.info("✓ SynergisticHybridGrader assembled")
        except Exception as exc:
            logger.error(f"✗ SynergisticHybridGrader failed: {exc}")
            _models["hybrid"] = None
    else:
        _models["hybrid"] = None

    # -- FeedbackGenerator (spaCy) ----------------------------------------
    try:
        from grading.feedback_generator import FeedbackGenerator  # noqa
        fg = FeedbackGenerator()
        _models["feedback"] = fg
        logger.info("✓ FeedbackGenerator ready")
    except Exception as exc:
        logger.warning(f"✗ FeedbackGenerator failed (fallback mode): {exc}")
        _models["feedback"] = None

    # -- HandwritingOCR (TrOCR) — lazy ------------------------------------
    _models["ocr"] = None  # Loaded on first /ocr call

    logger.info("=" * 50)
    logger.info("Phase 3 API ready on port 8002")
    logger.info("=" * 50)

    yield  # App runs here

    logger.info("Phase 3 API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Phase 3 — Synergistic Hybrid Grading API",
    description=(
        "Two-stage synergistic hybrid (SVM + Calibrated SBERT) "
        "with concept-level feedback and SHAP explainability. "
        "Backward-compatible with Phase 1/2 Node backend contract."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GradeRequest(BaseModel):
    question: str
    student_answer: str
    reference_answer: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Service health check.

    Returns model load status and fast-path threshold.
    """
    hybrid = _models.get("hybrid")
    path_stats = hybrid.path_stats() if hybrid else {}
    return {
        "status": "online",
        "phase": 3,
        "models": {
            "SVM":                 _models.get("classical") is not None,
            "SBERT_Calibrated":    _models.get("calibrated") is not None,
            "Synergistic_Hybrid":  hybrid is not None,
            "FeedbackGenerator":   _models.get("feedback") is not None,
        },
        "fast_path_threshold":    config.CONFIDENCE_THRESHOLD,
        "calibrated_thresholds":  (
            _models["calibrated"].thresholds
            if _models.get("calibrated") else None
        ),
        "path_stats": path_stats,
    }


@app.get("/")
def root():
    return {
        "status": "Phase 3 Synergistic Hybrid online",
        "phase": 3,
        "endpoints": ["/predict", "/ocr", "/explain", "/health"],
    }


@app.post("/predict")
async def predict(request: GradeRequest):
    """
    Grade a student answer against a reference.

    Two-stage synergistic hybrid:
      Stage 1 (fast path): SVM alone if confident (≥0.85)
      Stage 2 (ensemble):  SVM + Calibrated SBERT for uncertain cases

    Returns all fields required by Node backend (backward-compatible)
    plus Phase 3 concept-level extensions.

    Raises
    ------
    503 — models not loaded (run run_pipeline.py first)
    500 — internal grading error
    """
    hybrid  = _models.get("hybrid")
    feedgen = _models.get("feedback")

    if hybrid is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. Run: python phase3/run_pipeline.py "
                "to calibrate and save artefacts, then restart the API."
            ),
        )

    try:
        # Grade
        result = hybrid.grade(request.student_answer, request.reference_answer)

        # Concept feedback
        if feedgen:
            fb = feedgen.generate(
                request.student_answer,
                request.reference_answer,
                result["predicted_label"],
            )
        else:
            # Fallback when spaCy is unavailable
            fb = {
                "feedback": result["feedback"],
                "concepts_missing": [],
                "concepts_present": [],
                "completeness_score": 0.0,
                "topic_gap": "spaCy not available",
            }

        # Build response — all Phase 1/2 fields preserved
        return {
            # ── Node backend contract (required) ────────────────────
            "predicted_label":   result["predicted_label"],
            "similarity_score":  result["similarity_score"],   # None on fast path
            "confidence":        result["confidence"],
            "feedback":          fb["feedback"],
            # ── Phase 3 extensions ──────────────────────────────────
            "path":              result["path"],               # "fast" or "full"
            "stage":             result["stage"],              # 1 or 2
            "svm_confidence":    result["svm_confidence"],
            "concepts_missing":  fb["concepts_missing"],
            "concepts_present":  fb["concepts_present"],
            "completeness_score": fb["completeness_score"],
            "topic_gap":         fb["topic_gap"],
        }

    except Exception as exc:
        logger.error(f"/predict error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    Transcribe a handwritten answer image using TrOCR.

    Loads the TrOCR model lazily on first call (heavy download).
    Subsequent calls reuse the cached model.

    Returns
    -------
    {"transcribed_text": str}
    """
    # Lazy load TrOCR
    if _models.get("ocr") is None:
        try:
            sys.path.insert(0, _PH3)
            from phase3.ocr.trocr_engine import HandwritingOCR  # noqa
            _models["ocr"] = HandwritingOCR(model_name=config.TROCR_MODEL)
            logger.info("TrOCR model loaded on first /ocr call")
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"TrOCR model failed to load: {exc}. "
                    "Ensure transformers and torch are installed."
                ),
            )

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
        text     = _models["ocr"].transcribe(image)
        return {"transcribed_text": text.strip()}
    except Exception as exc:
        logger.error(f"/ocr error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/explain")
async def explain(
    student_answer: str  = Query(..., description="Student answer text"),
    reference_answer: str = Query(..., description="Reference answer text"),
    n: int               = Query(10, description="Number of top SHAP features to return"),
):
    """
    Return top-n SHAP feature attributions for a specific prediction.

    Uses coefficient-based linear approximation (fast, no SHAP overhead).
    Tells you which TF-IDF tokens most influenced the grade decision.

    Returns
    -------
    {"top_features": [{feature, shap_value, direction, tfidf}]}
    """
    classical = _models.get("classical")
    if classical is None:
        raise HTTPException(
            status_code=503,
            detail="ClassicalGrader not loaded. Run run_pipeline.py first.",
        )

    try:
        from evaluation.explainability import get_top_shap_features
        features = get_top_shap_features(
            classical, student_answer, reference_answer, n=n
        )
        return {"top_features": features}
    except Exception as exc:
        logger.error(f"/explain error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=False,
    )
