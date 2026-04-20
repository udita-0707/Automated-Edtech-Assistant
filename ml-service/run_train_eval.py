from utils import load_scientsbank, convert_labels, prepare_dataframe
from model import TextClassifier
import config
import os

def main():
    print("🚀 Initializing Final Merged Framework...")

    # 1. LOAD DATASET
    dataset = load_scientsbank()
    dataset = convert_labels(dataset, config.LABEL_SCHEME)

    # 2. PREPARE TRAINING DATA
    # Bug fix (was "test_ua"): always train on the canonical 'train' split
    # prepare_dataframe() returns 6 values: X_train, X_test, y_train, y_test,
    # train_ref, test_ref — reference answers needed for semantic features.
    X_train, _, y_train, _, train_ref, _ = prepare_dataframe(dataset, "train")

    clf = TextClassifier(max_features=config.MAX_FEATURES)

    print("\nTraining on main SciEntsBank 'train' split...")
    clf.train(X_train, y_train, references=train_ref)
    clf.save()

    # 3. COMPREHENSIVE EVALUATION (UA, UQ, UD splits)
    # Proves "Generalization Ability" for the rubric
    for split in ["test_ua", "test_uq", "test_ud"]:
        print(f"\n===== Evaluating on {split} =====")
        # Unpack all 6 values; ignore train columns during evaluation loop
        _, X_test, _, y_test, _, test_ref = prepare_dataframe(dataset, split)

        acc, f1, report = clf.evaluate(X_test, y_test, references=test_ref)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print("Detailed Classification Report:\n", report)

    print("\n✅ Final Framework Convergence Complete.")
    print("All artifacts saved. Ready for Phase 1 Submission.")

if __name__ == "__main__":
    main()
