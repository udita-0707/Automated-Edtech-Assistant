"""
Phase 3 Utilities
=================
Data loading and preprocessing helpers — functionally identical to
phase2/utils.py but imported from the phase3 namespace to keep the
two phases cleanly separated.

No modifications from Phase 2. The dataset pipeline is stable.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import load_dataset, ClassLabel

import config

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (idempotent)
# ---------------------------------------------------------------------------
for _resource in ("corpora/stopwords", "tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource.split("/")[1], quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_scientsbank():
    """
    Load the SciEntsBank dataset from HuggingFace Hub.

    Dataset: nkazi/SciEntsBank
    Splits: train, test_ua (unseen answers),
            test_uq (unseen questions),
            test_ud (unseen domains)

    Returns
    -------
    datasets.DatasetDict
    """
    return load_dataset(config.DATASET_NAME)


def convert_labels(dataset, scheme: str = config.LABEL_SCHEME):
    """
    Map the original 5-way SciEntsBank labels to a 3-way scheme.

    Original 5-way → 3-way mapping
    --------------------------------
    correct                       → 0  (correct)
    contradictory                 → 1  (contradictory)
    partially_correct_incomplete  → 2  (incorrect / other)
    irrelevant                    → 2
    non_domain                    → 2

    Rationale
    ---------
    The 5-way scheme has severe class imbalance and sparse signal in the
    fine-grained categories. Collapsing to 3-way improves learnability
    while preserving the grading semantics that matter pedagogically.

    Parameters
    ----------
    dataset : datasets.DatasetDict
    scheme  : str — only "3way" is supported

    Returns
    -------
    datasets.DatasetDict with remapped label column
    """
    if scheme == "3way":
        dataset = dataset.align_labels_with_mapping(
            {
                "correct": 0,
                "contradictory": 1,
                "partially_correct_incomplete": 2,
                "irrelevant": 2,
                "non_domain": 2,
            },
            "label",
        )
        dataset = dataset.cast_column(
            "label",
            ClassLabel(names=["correct", "contradictory", "incorrect"]),
        )
    return dataset


def prepare_dataframe(dataset, split: str):
    """
    Prepare aligned Series objects for a given evaluation split.

    Parameters
    ----------
    dataset : datasets.DatasetDict (output of convert_labels)
    split   : one of "train", "test_ua", "test_uq", "test_ud"

    Returns
    -------
    X_train    : pd.Series  — student answers from train
    X_test     : pd.Series  — student answers from split
    y_train    : pd.Series  — integer labels from train
    y_test     : pd.Series  — integer labels from split
    train_ref  : pd.Series  — reference answers from train
    test_ref   : pd.Series  — reference answers from split
    """
    train_df = dataset["train"].to_pandas()
    test_df  = dataset[split].to_pandas()
    return (
        train_df["student_answer"],
        test_df["student_answer"],
        train_df["label"],
        test_df["label"],
        train_df["reference_answer"],
        test_df["reference_answer"],
    )


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def get_jaccard_similarity(str1: str, str2: str) -> float:
    """
    Token-level Jaccard similarity after stopword removal.

    J(A,B) = |A ∩ B| / |A ∪ B|

    Stopwords are removed so content-bearing tokens drive the score.
    """
    a = {w for w in word_tokenize(str1.lower()) if w.isalnum() and w not in STOP_WORDS}
    b = {w for w in word_tokenize(str2.lower()) if w.isalnum() and w not in STOP_WORDS}
    if not a and not b:
        return 1.0
    return float(len(a & b)) / (len(a) + len(b) - len(a & b))


def get_token_density(student: str, reference: str) -> float:
    """
    Fraction of reference content tokens that appear in the student answer.

    density = |ref_tokens ∩ stu_tokens| / |ref_tokens|

    Measures recall of reference vocabulary — how many key terms did the
    student cover? Complementary to Jaccard which is symmetric.
    """
    ref_tokens = {w.lower() for w in word_tokenize(reference)
                  if w.isalnum() and w not in STOP_WORDS}
    stu_tokens = {w.lower() for w in word_tokenize(student)
                  if w.isalnum() and w not in STOP_WORDS}
    if not ref_tokens:
        return 1.0
    return len(ref_tokens & stu_tokens) / len(ref_tokens)
