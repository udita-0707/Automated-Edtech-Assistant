import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch

from grading.hybrid_grader import HybridGrader
from ocr.trocr_engine import HandwritingOCR
import config

app = FastAPI(title="Phase 2 Neural Grading API")

# Initialize models
# Lazy loading is better for dev, but for grading we load at startup
try:
    grader = HybridGrader(alpha=config.HYBRID_ALPHA)
    grader.load(config.MODEL_A_PATH)
    
    ocr_engine = HandwritingOCR(model_name=config.TROCR_MODEL)
except Exception as e:
    print(f"Warning: Models failed to load. Ensure training is run first. Error: {e}")

class GradeRequest(BaseModel):
    student_answer: str
    reference_answer: str

@app.get("/")
def root():
    return {"status": "Phase 2 Pipeline Online", "modes": ["Hybrid Grading", "Neural OCR"]}

@app.post("/predict")
def predict(request: GradeRequest):
    """
    Grades an answer using the Hybrid Model C (SVM + SBERT).
    """
    try:
        result = grader.grade(request.student_answer, request.reference_answer)
        
        label_map = {0: "correct", 1: "incorrect", 2: "partially correct"}
        
        # Add feedback logic similar to Phase 1 for consistency
        label_to_feedback = {
            0: "Excellent. Semantic matching confirms a near-perfect conceptual overlap.",
            1: "Conceptual mismatch detected. The answer contradicts stored reference domain knowledge.",
            2: "Partial conceptual overlap. While keywords are present, semantic depth is lacking."
        }
        
        return {
            "predicted_label": label_map.get(result["label_idx"], "incorrect"),
            "confidence": result["confidence"],
            "similarity_score": result["similarity_score"],
            "feedback": label_to_feedback.get(result["label_idx"], "Improve conceptual clarity.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    Transcribes handwriting using the TrOCR Transformer model.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        text = ocr_engine.transcribe(image)
        return {"transcribed_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
