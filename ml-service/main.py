from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import TextClassifier
import config
import os
import numpy as np

app = FastAPI(title="Automated EdTech Grading Assistant - ML Service")

# Initialize and load the unified classifier
predictor = TextClassifier()
if not predictor.load():
    print("⚠️ No pre-trained model found. Please run run_train_eval.py first or model will be untrained.")

class GradeRequest(BaseModel):
    question: str
    student_answer: str
    reference_answer: str

@app.get("/")
async def root():
    return {
        "status": "online",
        "model": "Merged Semantic Regression",
        "version": "1.2.0 (Advanced Features)"
    }

@app.post("/predict")
async def predict(request: GradeRequest):
    try:
        # Get detailed metrics from the classifier
        details = predictor.predict_detailed(request.student_answer, request.reference_answer)
        
        # Mapping our 3-way labels to Frontend expectations:
        # 0: correct -> 'correct'
        # 1: contradictory -> 'incorrect'
        # 2: incorrect/partial -> 'partially correct' (since it includes partially_correct_incomplete)
        
        label_map = {
            0: "correct",
            1: "incorrect",
            2: "partially correct"
        }
        
        return {
            "predicted_label": label_map.get(details["label_idx"], "incorrect"),
            "similarity_score": details["similarity_score"],
            "length_ratio": details["length_ratio"],
            "confidence": details["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
