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
    Strategic Dataset Binning
    Maps 5-way labels to 3-way based on academic grading standards:
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

def prepare_dataframe(dataset, split):
    train_df = dataset["train"].to_pandas()
    test_df = dataset[split].to_pandas()
    return train_df["student_answer"], test_df["student_answer"], train_df["label"], test_df["label"]
