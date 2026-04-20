import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_scientsbank, convert_labels, prepare_dataframe
from grading.classical_grader import ClassicalGrader
from grading.semantic_scorer import SemanticScorer
from grading.hybrid_grader import HybridGrader

LABEL_NAMES = ['correct', 'contradictory', 'incorrect']
OUT_DIR = "phase2/evaluation"


def plot_model_comparison(all_results, out_dir):
    """Grouped bar chart comparing F1 across models and splits."""
    df = pd.DataFrame(all_results)
    models = df["model"].unique()
    splits = df["split"].unique()

    x = np.arange(len(splits))
    width = 0.25
    colors = ['#E67E22', '#27AE60', '#8E44AD']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        vals = df[df["model"] == model].set_index("split").loc[splits, "f1_macro"].values
        bars = ax.bar(x + i * width, vals, width, label=model, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Test Split')
    ax.set_ylabel('F1 Macro')
    ax.set_title('Phase 2 Model Comparison Across Splits')
    ax.set_xticks(x + width)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=150)
    plt.close()
    print("  ✅ Model comparison chart saved.")


def main():
    print("🚀 Initializing Phase 2 Neural/Hybrid Pipeline...")

    dataset = load_scientsbank()
    dataset = convert_labels(dataset, "3way")

    X_train, _, y_train, _, _, _ = prepare_dataframe(dataset, "train")
    print(f"Training Model A (SVM) on {len(X_train)} samples...")
    model_a = ClassicalGrader()
    model_a.train(X_train, y_train)
    model_a_path = "phase2/data/artifacts/model_a.pkl"
    os.makedirs(os.path.dirname(model_a_path), exist_ok=True)
    model_a.save(model_a_path)

    scorer_b = SemanticScorer()
    hybrid_c = HybridGrader(alpha=0.4)
    hybrid_c.load(model_a_path)

    splits = ["test_ua", "test_uq", "test_ud"]
    all_results = []

    os.makedirs(OUT_DIR, exist_ok=True)

    model_configs = [
        ("Model A (SVM)", "A"),
        ("Model B (SBERT)", "B"),
        ("Model C (Hybrid)", "C"),
    ]

    for split in splits:
        print(f"\n📊 Evaluating on {split}...")
        _, X_test, _, y_test, _, test_ref = prepare_dataframe(dataset, split)

        y_true = y_test.tolist()
        predictions = {"A": [], "B": [], "C": []}

        for i in range(len(X_test)):
            stu = X_test.iloc[i]
            ref = test_ref.iloc[i]
            predictions["A"].append(model_a.predict(stu))
            predictions["B"].append(scorer_b.grade(stu, ref))
            predictions["C"].append(hybrid_c.grade(stu, ref)["label_idx"])

        for mode_name, mode_key in model_configs:
            y_pred = predictions[mode_key]
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            prec = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')

            all_results.append({
                "split": split,
                "model": mode_name,
                "accuracy": round(acc, 4),
                "precision_macro": round(prec, 4),
                "recall_macro": round(rec, 4),
                "f1_macro": round(f1, 4),
            })
            print(f"  {mode_name}: Acc={acc:.4f}, F1={f1:.4f}")

            # Full classification report
            report_dict = classification_report(
                y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.to_csv(
                os.path.join(OUT_DIR, f"classification_report_{mode_key}_{split}.csv")
            )

            # Confusion matrices for test_ua
            if split == "test_ua":
                cm = confusion_matrix(y_true, y_pred)
                cmap_map = {"A": "Oranges", "B": "Purples", "C": "Greens"}
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_map[mode_key],
                            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
                plt.title(f'{mode_name} Confusion Matrix (test_ua)')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f"conf{mode_key}.png"), dpi=150)
                plt.close()
                print(f"    ✅ Confusion matrix saved (conf{mode_key}.png)")

    # ── Alpha Calibration ──
    print("\n🔨 Generating Alpha Calibration Plot...")
    alphas = np.linspace(0.0, 1.0, 11)
    f1_scores = []
    _, X_val, _, y_val, _, val_ref = prepare_dataframe(dataset, "test_ua")
    y_true_val = y_val.tolist()

    for a in alphas:
        hybrid_temp = HybridGrader(alpha=a)
        hybrid_temp.load(model_a_path)
        y_pred = []
        for i in range(len(X_val)):
            stu = X_val.iloc[i]
            ref = val_ref.iloc[i]
            y_pred.append(hybrid_temp.grade(stu, ref)["label_idx"])
        f1_scores.append(f1_score(y_true_val, y_pred, average='macro'))

    best_alpha = alphas[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, f1_scores, marker='o', linestyle='-', color='#8E44AD', linewidth=2)
    plt.axvline(x=best_alpha, color='red', linestyle='--', alpha=0.7,
                label=f'Best α={best_alpha:.1f} (F1={best_f1:.3f})')
    plt.title('Hybrid Ensemble Alpha Calibration (Validation Set)')
    plt.xlabel('Alpha (SVM Weight)')
    plt.ylabel('F1 Score (Macro)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "alpha_calibration.png"), dpi=150)
    plt.close()
    print(f"  ✅ Alpha calibration plot saved (best α={best_alpha:.1f}).")

    # ── Model comparison chart ──
    print("\n📊 Generating model comparison chart...")
    plot_model_comparison(all_results, OUT_DIR)

    # ── Save CSV + JSON ──
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "ablation_phase2.csv"), index=False)

    metrics_summary = {
        "phase": 2,
        "models": ["SVM+TF-IDF", "SBERT (all-MiniLM-L6-v2)", "Hybrid Ensemble"],
        "best_alpha": round(float(best_alpha), 2),
        "alpha_f1_scores": [round(float(f), 4) for f in f1_scores],
        "results": all_results,
    }
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n✅ Phase 2 Evaluation Complete. Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
