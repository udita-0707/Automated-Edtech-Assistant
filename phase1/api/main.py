import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import TextClassifier
import config

# ---------------------------------------------------------------------------
# Try to import pytesseract for the /ocr endpoint.
# If not installed the endpoint will return a 503 instead of crashing startup.
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

app = FastAPI(title="Automated EdTech Grading Assistant - ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize and load the unified classifier
predictor = TextClassifier()
if not predictor.load():
    print("⚠️  No pre-trained model found. Run run_train_eval.py first.")


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

@app.get("/")
async def root():
    return {
        "status": "online",
        "model": "Merged Semantic Regression",
        "version": "1.3.0 (Phase 1 — feedback + OCR)"
    }


@app.post("/predict")
async def predict(request: GradeRequest):
    """
    Grade a student answer against a reference answer.

    Returns:
      predicted_label: 'correct' | 'incorrect' | 'partially correct'
      similarity_score: Jaccard overlap (0–1)
      length_ratio: student/reference length ratio
      confidence: model confidence (0–1)
      feedback: human-readable feedback string
    """
    try:
        details = predictor.predict_detailed(
            request.student_answer, request.reference_answer
        )

        # Map 3-way integer labels to UI-friendly strings
        # 0: correct, 1: contradictory → incorrect, 2: incorrect → partially correct
        label_map = {
            0: "correct",
            1: "incorrect",
            2: "partially correct",
        }

        return {
            "predicted_label": label_map.get(details["label_idx"], "incorrect"),
            "similarity_score": details["similarity_score"],
            "length_ratio": details["length_ratio"],
            "confidence": details["confidence"],
            "feedback": details["feedback"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    Transcribe a handwritten answer image using Tesseract OCR.

    OCR pipeline:
      1. Grayscale conversion (removes colour noise)
      2. Auto-contrast (CLAHE-equivalent via PIL ImageOps)
      3. Sharpen filter (boosts pen stroke edges)
      4. Binarisation via lambda threshold
      5. Tesseract --oem 0 --psm 6 (legacy engine, uniform block)
         --oem 0 forces the original Tesseract cube engine (non-neural),
         which is faster and robust on binarised handwriting.
         --oem 3 (LSTM) is better on printed text but slower here.

    Returns:
      transcribed_text: cleaned OCR output string
    """
    if not _TESSERACT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="pytesseract / Pillow not installed. Run: pip install pytesseract Pillow"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Pre-processing pipeline
        gray   = image.convert("L")                          # step 1: grayscale
        ac     = ImageOps.autocontrast(gray, cutoff=1)       # step 2: auto-contrast
        sharp  = ac.filter(ImageFilter.SHARPEN)              # step 3: sharpen
        binary = sharp.point(lambda p: 255 if p > 128 else 0, "1")  # step 4: binarise

        custom_config = "--oem 0 --psm 6"
        raw_text = pytesseract.image_to_string(binary, config=custom_config)

        # Clean: strip leading/trailing whitespace, collapse internal newlines
        cleaned = " ".join(raw_text.split())

        return {"transcribed_text": cleaned}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
