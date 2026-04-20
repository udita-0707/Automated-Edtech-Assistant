from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
import joblib
import os

class ClassicalGrader:
    """
    Model A: Classical Baseline for Phase 2.
    Uses TF-IDF vectorization with an RBF-kernel SVM.
    
    Theoretical Context:
    The RBF kernel allows for non-linear decision boundaries in a 
    high-dimensional feature space, which is more robust for short 
    answer grading than linear regression.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.model = SVC(kernel='rbf', probability=True, class_weight='balanced')

    def train(self, X, y):
        X_tfidf = self.vectorizer.fit_transform(X)
        self.model.fit(X_tfidf, y)

    def predict(self, text):
        X_tfidf = self.vectorizer.transform([text])
        return self.model.predict(X_tfidf)[0]

    def predict_probs(self, text):
        X_tfidf = self.vectorizer.transform([text])
        return self.model.predict_proba(X_tfidf)[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'v': self.vectorizer, 'm': self.model}, path)

    def load(self, path):
        data = joblib.load(path)
        self.vectorizer = data['v']
        self.model = data['m']
