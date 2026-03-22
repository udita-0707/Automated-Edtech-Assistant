# % # ANALYTIC REPORT: Automated EdTech Grading Assistant
# # Advanced Dataset Characterization & Manifold Analysis
# # =========================================================
# 
# This notebook documents the core analytical framework and advanced manifold
# justifications for our multi-dimensional semantic feature pipeline.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add ml-service to path to import unified utils
sys.path.append(os.path.abspath("../ml-service"))
from utils import load_scientsbank, convert_labels, get_jaccard_similarity, get_token_density
import config

def perform_eda(dataset):
    print("\n================= EDA START =================")
    train_df = dataset["train"].to_pandas()

    # 1. BASIC INFO
    print("\n--- 1. Dataset Shape ---")
    print(train_df.shape)

    # 2. LABEL DISTRIBUTION
    print("\n--- 2. Label Distribution ---")
    label_counts = train_df["label"].value_counts()
    print(label_counts)
    label_counts.plot(kind="bar", color=['#4ade80', '#fbbf24', '#f87171'])
    plt.title("Label Distribution")
    plt.savefig("label_distribution.png")
    plt.clf()
    print("[INSIGHT] Class imbalance exists. Majority class dominates → model bias risk.")

    # 3. TEXT LENGTH ANALYSIS
    train_df["text_len"] = train_df["student_answer"].apply(len)
    print("\n--- 3. Text Length Stats ---")
    print(train_df["text_len"].describe())
    plt.hist(train_df["text_len"], bins=50)
    plt.title("Text Length Distribution")
    plt.savefig("text_length_distribution.png")
    plt.clf()

    # 4. CLASS-WISE TEXT LENGTH
    print("\n--- 4. Avg Length per Class ---")
    print(train_df.groupby("label")["text_len"].mean())
    train_df.boxplot(column="text_len", by="label")
    plt.title("Text Length by Class")
    plt.savefig("text_length_by_class.png")
    plt.clf()
    print("[INSIGHT] Variation in answer length across classes → useful feature for classification.")

    # 5. WORD COUNT ANALYSIS
    train_df["word_count"] = train_df["student_answer"].apply(lambda x: len(x.split()))
    print("\n--- 5. Word Count Stats ---")
    print(train_df["word_count"].describe())

    # 6. DUPLICATES CHECK
    duplicates = train_df.duplicated(subset=["student_answer"]).sum()
    print("\n--- 6. Duplicate Answers ---")
    print("Duplicates:", duplicates)
    print("[INSIGHT] Duplicate responses may introduce bias and reduce model generalization.")

    # 7. SPLIT COMPARISON 🔥 CRITICAL
    print("\n--- 7. Split Size Comparison ---")
    for split in dataset:
        print(split, ":", len(dataset[split]))
    print("[INSIGHT] Dataset contains multiple evaluation settings (UA, UQ, UD) → introduces distribution shift.")

    # 8. CLASS IMBALANCE RATIO
    print("\n--- 8. Class Imbalance Ratio ---")
    print(train_df["label"].value_counts(normalize=True))

    # 9. TEXT UNIQUENESS
    unique_ratio = train_df["student_answer"].nunique() / len(train_df)
    print("\n--- 9. Unique Answer Ratio ---")
    print("Unique Ratio:", unique_ratio)
    print("[INSIGHT] High uniqueness indicates diverse language usage → harder NLP task.")

    # 10. SHORT vs LONG ANSWERS
    short_answers = (train_df["word_count"] < 5).sum()
    long_answers = (train_df["word_count"] > 20).sum()
    print("\n--- 10. Answer Length Categories ---")
    print("Short Answers (<5 words):", short_answers)
    print("Long Answers (>20 words):", long_answers)
    print("[INSIGHT] Presence of variability in response style indicates diverse student population.")

    print("\n================= EDA END =================")

# ---------------------------
# Execution Logic
# ---------------------------
ds = load_scientsbank()
ds = convert_labels(ds, config.LABEL_SCHEME)
perform_eda(ds)

# ---------------------------
# 11. ADVANCED FEATURE MANIFOLD
# ---------------------------
print("\n--- 11. Running Advanced Feature Extraction ---")
train_df = ds["train"].to_pandas()
sample_size = 400
X_list = []
y_list = []

for i in range(sample_size):
    row = train_df.iloc[i]
    jaccard = get_jaccard_similarity(row['student_answer'], row['reference_answer'])
    density = get_token_density(row['student_answer'], row['reference_answer'])
    len_ratio = len(row['student_answer']) / max(len(row['reference_answer']), 1)
    X_list.append([jaccard, density, len_ratio])
    y_list.append(row['label'])

X = np.array(X_list)
y = np.array(y_list)

# ---------------------------
# 12. PCA VISUAL PROOF
# ---------------------------
print("--- 12. Calculating PCA Projection ---")
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap='Set1', alpha=0.8)
plt.colorbar(scatter, label='Class (0:Correct, 1:Contradictory, 2:Incorrect)')
plt.title('PCA Projection of Semantic Manifold')
plt.savefig('pca_visual_proof.png')
plt.clf()

print("\n=========================================================")
print("FINAL EDA COMPLETE: 12 Sections of Insights Generated.")
print("=========================================================")
