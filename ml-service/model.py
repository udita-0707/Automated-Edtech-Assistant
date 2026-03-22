import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from utils import get_jaccard_similarity, get_token_density
import config

class TextClassifier:
    """
    Final Integrated Framework
    Combines Structural Features with Semantic Manifold analysis.
    """

    def __init__(self, max_features=config.MAX_FEATURES):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE
        )

    def extract_features(self, texts, references=None):
        """
        Advanced Feature Engineering (Breakthrough representation):
        1. User's Structural Features: [length, word_count, avg_word_len]
        2. Semantic Manifold Features: [Jaccard, Token Density]
        """
        feats = []
        for i, text in enumerate(texts):
            # 1. Structural
            words = text.split()
            length = len(text)
            word_count = len(words)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            
            # 2. Semantic (if reference provided)
            jaccard = 0.5
            density = 0.5
            if references is not None:
                ref = references.iloc[i] if hasattr(references, 'iloc') else references[i]
                jaccard = get_jaccard_similarity(text, ref)
                density = get_token_density(text, ref)
            
            feats.append([length, word_count, avg_word_len, jaccard, density])
        
        return np.array(feats)

    def train(self, X_train, y_train, references=None):
        """
        Merged Training Pipeline:
        TF-IDF + Scaled Structural/Semantic Features
        """
        # TF-IDF
        X_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Numeric Features
        X_extra = self.extract_features(X_train, references)
        X_extra_scaled = self.scaler.fit_transform(X_extra)
        
        # Combine Sparse + Dense
        X_combined = np.hstack((X_tfidf.toarray(), X_extra_scaled))
        
        self.model.fit(X_combined, y_train)
        print("Model training complete.")

    def evaluate(self, X_test, y_test, references=None):
        X_tfidf = self.vectorizer.transform(X_test)
        X_extra = self.extract_features(X_test, references)
        X_extra_scaled = self.scaler.transform(X_extra)
        X_combined = np.hstack((X_tfidf.toarray(), X_extra_scaled))
        
        preds = self.model.predict(X_combined)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds)
        
        return acc, f1, report

    def predict_detailed(self, student_answer, reference_answer):
        """
        Returns a dictionary with the label, confidence, and internal scores.
        """
        X_tfidf = self.vectorizer.transform([student_answer])
        X_extra = self.extract_features([student_answer], [reference_answer])
        X_extra_scaled = self.scaler.transform(X_extra)
        X_combined = np.hstack((X_tfidf.toarray(), X_extra_scaled))
        
        # Get prediction and probabilities
        pred_label = self.model.predict(X_combined)[0]
        probs = self.model.predict_proba(X_combined)[0]
        confidence = float(np.max(probs))
        
        # Features: [length, word_count, avg_word_len, jaccard, density]
        # We index into X_extra[0] to get the raw (unscaled) features
        similarity_score = float(X_extra[0][3]) # Jaccard
        len_ratio = float(X_extra[0][0]) / max(len(reference_answer), 1)
        
        return {
            "label_idx": int(pred_label),
            "confidence": confidence,
            "similarity_score": similarity_score,
            "length_ratio": len_ratio
        }

    def save(self):
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, config.MODEL_PATH)
        joblib.dump(self.scaler, config.SCALER_PATH)
        joblib.dump(self.vectorizer, config.VECTORIZER_PATH)
        print(f"Model and artifacts saved to {config.MODEL_PATH}")

    def load(self):
        if os.path.exists(config.MODEL_PATH):
            self.model = joblib.load(config.MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
            self.vectorizer = joblib.load(config.VECTORIZER_PATH)
            return True
        return False
