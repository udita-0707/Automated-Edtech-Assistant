# Automated EdTech Grading Assistant

![Project Architecture](https://img.shields.io/badge/Architecture-React%20%7C%20Node%20%7C%20FastAPI-blue.svg)
![ML Logic](https://img.shields.io/badge/Machine%20Learning-TF--IDF%20%7C%20Logistic%20Regression-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-SciEntsBank-orange.svg)

## Problem Motivation
Evaluating student short-text responses is traditionally qualitative and subjective. The **Automated EdTech Grading Assistant** provides an algorithmic quantitative assessment using a specialized Machine Learning pipeline. Our solution focuses on **Explainable AI (XAI)** by utilizing clear mathematical boundaries to detect academic correctness, rather than functioning as a black-box large language model.

## Method & Approach
This repository implements a rigorous Machine Learning pipeline for automated grading:

1. **Dataset Analysis & Preprocessing**: We utilized the `nkazi/SciEntsBank` dataset containing over 4,900 academic records. Our framework achieves **~62% Accuracy** on unseen evaluation splits using a strategic 3-way categorical reduction.
2. **Feature Engineering**: 
   - **Structural Metrics**: Word Count, Answer Length, Avg Word Length.
   - **Semantic Features**: Jaccard Similarity and Token Match Density.
   - **N-gram Representation**: TF-IDF Bigrams for local contextual cues.
3. **Model Selection**: A transparent `LogisticRegression` classifier with `class_weight="balanced"` to handle dataset imbalance.
4. **Clean Code & Modularity**: A production-ready 3-tier architecture with centralized configuration (`config.py`).

---

## Repository Structure

```
Automated-Edtech-Assistant/
├── frontend/             # React + Vite + Tailwind v4 Web Application
├── backend/              # Node.js + Express API Gateway
├── ml-service/           # Python FastAPI ML Microservice
│   ├── data/             # Persistent Model Binaries
│   ├── main.py           # API Controller
│   ├── model.py          # ML Core & Vectorization
│   ├── utils.py          # NLP Heuristics & EDA Logic
│   ├── config.py         # Hyperparameters & Paths
│   ├── requirements.txt  # Dependencies
│   └── README.md         # Service Documentation
├── notebooks/            # Data Visualizations & Independent EDA
├── report/               # Academic Project Documentation (LaTeX)
└── README.md             # Project Overview
```

---

## Setup & Reproducibility 

### 1. ML Backend (FastAPI)
```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
> **Note**: On the first genesis, the model dynamically caches and serializes its state into `data/model.pkl` after streaming the SciEntsBank dataset.

### 2. Node Backend API
```bash
cd backend
npm install
node index.js
```
The Express backend manages the `database.sqlite` and orchestrates HTTP transactions.

### 3. React Frontend
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5174` in your browser.

---

## Results & Demo
- Real-time submission validation.
- SQLite-powered persistent grading history.
- Statistical breakdown visualizations mapped accurately across the SciEntsBank labeling matrix.

### Failure Analysis (Limitations)
As demonstrated theoretically, TF-IDF cannot solve advanced syntactic negations. E.g., *"It is not a cell"* shares heavy token overlap with *"It is a cell"*. Future iterations in Phase 2 will implement deep contextual embeddings (`Sentence-Transformers`) to resolve syntactic contradiction logic.
