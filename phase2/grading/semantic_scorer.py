import torch
from sentence_transformers import SentenceTransformer, util

class SemanticScorer:
    """
    Model B: Semantic Similarity Scorer.
    Uses SBERT (all-MiniLM-L6-v2) to compute cosine similarity.
    
    Theoretical Context:
    Sentence BERT uses Siamese networks to create semantically 
    meaningful embeddings. Cosine similarity in this embedding space 
    captures relationships that TF-IDF (keyword-based) misses.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)

    def score(self, student_answer, reference_answer):
        """
        Returns a cosine similarity score in [0, 1].
        """
        emb1 = self.model.encode(student_answer, convert_to_tensor=True)
        emb2 = self.model.encode(reference_answer, convert_to_tensor=True)
        
        cos_sim = util.cos_sim(emb1, emb2)
        return float(cos_sim.item())

    def grade(self, student_answer, reference_answer):
        """
        Heuristic grading based on similarity thresholds:
        - Correct: > 0.75
        - Partial: 0.45 - 0.75
        - Incorrect: < 0.45
        """
        score = self.score(student_answer, reference_answer)
        if score > 0.75:
            return 0 # correct
        elif score > 0.45:
            return 2 # partial
        else:
            return 1 # incorrect (contradictory/poor)
