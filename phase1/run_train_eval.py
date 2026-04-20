import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    classification_report, precision_score, recall_score
)
from utils import load_scientsbank, convert_labels, prepare_dataframe
from grading.model import TextClassifier
import config
import os

LABEL_NAMES = ['correct', 'contradictory', 'incorrect']
OUT_DIR = "phase1/evaluation"


def plot_label_distribution(dataset, out_dir):
    """Bar chart showing class counts across train + all test splits."""
    splits = ["train", "test_ua", "test_uq", "test_ud"]
    counts = {s: {} for s in splits}
    for s in splits:
        df = dataset[s].to_pandas()
        vc = df["label"].value_counts().sort_index()
        for idx, name in enumerate(LABEL_NAMES):
            counts[s][name] = int(vc.get(idx, 0))

    x = np.arange(len(LABEL_NAMES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for i, s in enumerate(splits):
        vals = [counts[s].get(n, 0) for n in LABEL_NAMES]
        ax.bar(x + i * width, vals, width, label=s, color=colors[i])
    ax.set_xlabel('Label')
    ax.set_ylabel('Sample Count')
    ax.set_title('Label Distribution Across Splits')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(LABEL_NAMES)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_distribution.png"), dpi=150)
    plt.close()
    print("  ✅ Label distribution plot saved.")
    return counts


def plot_feature_importance(clf, out_dir, top_n=20):
    """Show top TF-IDF feature coefficients from the LogReg model."""
    feature_names = clf.vectorizer.get_feature_names_out()
    # Average absolute coefficient across classes
    coefs = np.abs(clf.model.coef_).mean(axis=0)
    # Only TF-IDF features (first len(feature_names) columns)
    tfidf_coefs = coefs[:len(feature_names)]
    top_idx = np.argsort(tfidf_coefs)[-top_n:]
    top_feats = feature_names[top_idx]
    top_vals = tfidf_coefs[top_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), top_vals, color='#4C72B0')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.set_xlabel('Mean |Coefficient|')
    ax.set_title(f'Top {top_n} TF-IDF Feature Importances (LogReg)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print("  ✅ Feature importance plot saved.")


def plot_accuracy_comparison(results, out_dir):
    """Bar chart comparing accuracy across splits."""
    splits = [r["split"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1_macro"] for r in results]

    x = np.arange(len(splits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, accs, width, label='Accuracy', color='#4C72B0')
    ax.bar(x + width / 2, f1s, width, label='F1 Macro', color='#C44E52')
    ax.set_ylabel('Score')
    ax.set_title('Phase 1 Performance Across Test Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1)
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width / 2, a + 0.02, f'{a:.3f}', ha='center', fontsize=8)
        ax.text(i + width / 2, f + 0.02, f'{f:.3f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  ✅ Accuracy comparison plot saved.")


def main():
    print("🚀 Initializing Phase 1 Evaluation Pipeline...")

    dataset = load_scientsbank()
    dataset = convert_labels(dataset, config.LABEL_SCHEME)

    X_train, _, y_train, _, train_ref, _ = prepare_dataframe(dataset, "train")

    clf = TextClassifier(max_features=config.MAX_FEATURES)

    print(f"\nTraining on {len(X_train)} SciEntsBank 'train' samples...")
    clf.train(X_train, y_train, references=train_ref)
    clf.save()

    splits = ["test_ua", "test_uq", "test_ud"]
    all_results = []

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Label distribution ──
    print("\n📊 Generating label distribution plot...")
    label_counts = plot_label_distribution(dataset, OUT_DIR)

    # ── Per-split evaluation ──
    for split in splits:
        print(f"\n===== Evaluating on {split} =====")
        _, X_test, _, y_test, _, test_ref = prepare_dataframe(dataset, split)

        X_tfidf = clf.vectorizer.transform(X_test)
        X_extra = clf.extract_features(X_test, test_ref)
        X_extra_scaled = clf.scaler.transform(X_extra)
        X_combined = np.hstack((X_tfidf.toarray(), X_extra_scaled))

        preds = clf.model.predict(X_combined)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Macro:  {f1:.4f}")

        # Full classification report
        report_str = classification_report(
            y_test, preds, target_names=LABEL_NAMES, output_dict=False
        )
        report_dict = classification_report(
            y_test, preds, target_names=LABEL_NAMES, output_dict=True
        )
        print(report_str)

        # Save per-class report CSV
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(os.path.join(OUT_DIR, f"classification_report_{split}.csv"))

        all_results.append({
            "split": split,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
        })

        # ── Confusion matrix (for every split) ──
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        plt.title(f'Phase 1 Confusion Matrix ({split})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        fname = "confusion_matrix.png" if split == "test_ua" else f"confusion_matrix_{split}.png"
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close()
        print(f"  ✅ Confusion matrix saved ({fname}).")

    # ── Feature importance ──
    print("\n📊 Generating feature importance plot...")
    plot_feature_importance(clf, OUT_DIR, top_n=20)

    # ── Accuracy comparison ──
    print("📊 Generating accuracy comparison plot...")
    plot_accuracy_comparison(all_results, OUT_DIR)

    # ── Save CSV + JSON ──
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "ablation_phase1.csv"), index=False)

    metrics_summary = {
        "phase": 1,
        "model": "LogReg + TF-IDF + Jaccard + TokenDensity",
        "label_counts": label_counts,
        "results": all_results,
    }
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n✅ Phase 1 Evaluation Complete.")
    print(f"All artifacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
