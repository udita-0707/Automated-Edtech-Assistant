import numpy as np
from .classical_grader import ClassicalGrader
from .semantic_scorer import SemanticScorer

class HybridGrader:
    """
    Model C: Hybrid Ensemble Grader.
    Ensembles Model A (SVM) and Model B (SBERT).
    
    Theoretical Context:
    By combining keyword-based structural analysis (SVM) with 
    deep semantic embeddings (SBERT), we mitigate the 'keyword matching' 
    limitations of classical models and the 'hallucination' risks 
    of pure semantic models. 
    Formula: Final_Score = α * P(Model_A) + (1-α) * P(Model_B)
    """
    
    def __init__(self, alpha=0.4):
        self.model_a = ClassicalGrader()
        self.model_b = SemanticScorer()
        self.alpha = alpha

    def load(self, model_a_path):
        self.model_a.load(model_a_path)

    def grade(self, student_answer, reference_answer):
        """
        Computes a hybrid grade.
        1. Get SVM probabilities (Model A)
        2. Get SBERT similarity (Model B)
        3. Simple weighted ensemble of probabilities.
        """
        # Model A: SVM Probs [P(correct), P(contradictory), P(partial)]
        probs_a = self.model_a.predict_probs(student_answer)
        
        # Model B: SBERT Score mapped to pseudo-probabilities
        score_b = self.model_b.score(student_answer, reference_answer)
        
        # Mapping SBERT score to 3-way distribution (heuristic)
        # Correct if high sim, Incorrect if low, Partial if mid
        if score_b > 0.8:
            probs_b = np.array([0.9, 0.05, 0.05])
        elif score_b < 0.4:
            probs_b = np.array([0.05, 0.9, 0.05])
        else:
            probs_b = np.array([0.1, 0.1, 0.8])
            
        # Weighted ensemble
        final_probs = self.alpha * probs_a + (1 - self.alpha) * probs_b
        label_idx = np.argmax(final_probs)
        
        return {
            "label_idx": int(label_idx),
            "confidence": float(np.max(final_probs)),
            "similarity_score": score_b,
            "components": {
                "svm_probs": probs_a.tolist(),
                "sbert_sim": score_b
            }
        }
