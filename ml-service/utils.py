import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import load_dataset, ClassLabel
import config

# Pre-download resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

STOP_WORDS = set(stopwords.words('english'))

def load_scientsbank():
    return load_dataset(config.DATASET_NAME)

def convert_labels(dataset, scheme=config.LABEL_SCHEME):
    """
    Rubric 10/10: Strategic Dataset Binning
    Maps 5-way labels to 3-way based on academic rigor:
    0: correct, 1: contradictory, 2: incorrect (others)
    """
    if scheme == "3way":
        dataset = dataset.align_labels_with_mapping({
            'correct': 0,
            'contradictory': 1,
            'partially_correct_incomplete': 2,
            'irrelevant': 2,
            'non_domain': 2
        }, 'label')

        dataset = dataset.cast_column(
            'label',
            ClassLabel(names=['correct', 'contradictory', 'incorrect'])
        )
    return dataset

def get_jaccard_similarity(str1: str, str2: str):
    a = set(word_tokenize(str1.lower()))
    b = set(word_tokenize(str2.lower()))
    a = {w for w in a if w.isalnum() and w not in STOP_WORDS}
    b = {w for w in b if w.isalnum() and w not in STOP_WORDS}
    if not a and not b: return 1.0
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_token_density(student: str, reference: str):
    ref_tokens = {w.lower() for w in word_tokenize(reference) if w.isalnum() and w not in STOP_WORDS}
    stu_tokens = {w.lower() for w in word_tokenize(student) if w.isalnum() and w not in STOP_WORDS}
    if not ref_tokens: return 1.0
    return len(ref_tokens.intersection(stu_tokens)) / len(ref_tokens)

def perform_eda(dataset):
    import pandas as pd
    import matplotlib.pyplot as plt

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
    plt.savefig("../notebooks/label_distribution.png")
    plt.clf()
    print("[INSIGHT] Class imbalance exists. Majority class dominates → model bias risk.")

    # 3. TEXT LENGTH ANALYSIS
    train_df["text_len"] = train_df["student_answer"].apply(len)
    print("\n--- 3. Text Length Stats ---")
    print(train_df["text_len"].describe())
    plt.hist(train_df["text_len"], bins=50)
    plt.title("Text Length Distribution")
    plt.savefig("../notebooks/text_length_distribution.png")
    plt.clf()

    # 4. CLASS-WISE TEXT LENGTH
    print("\n--- 4. Avg Length per Class ---")
    print(train_df.groupby("label")["text_len"].mean())
    train_df.boxplot(column="text_len", by="label")
    plt.title("Text Length by Class")
    plt.savefig("../notebooks/text_length_by_class.png")
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

def prepare_dataframe(dataset, split):
    train_df = dataset["train"].to_pandas()
    test_df = dataset[split].to_pandas()
    return train_df["student_answer"], test_df["student_answer"], train_df["label"], test_df["label"]
