import pandas as pd
from utils import load_scientsbank, convert_labels, prepare_dataframe
from grading.model import TextClassifier
import config

def run_phase1_ablation():
    """
    Runs the Phase 1 ablation study.
    Evaluates the classical baseline model across all SciEntsBank test splits.
    """
    print("📊 Starting Phase 1 Ablation Study...")
    
    dataset = load_scientsbank()
    dataset = convert_labels(dataset, config.LABEL_SCHEME)
    
    # Initialize classifier
    clf = TextClassifier()
    
    # 1. Train on standard train set
    X_train, _, y_train, _, train_ref, _ = prepare_dataframe(dataset, "train")
    print(f"Training on {len(X_train)} samples...")
    clf.train(X_train, y_train, references=train_ref)
    
    results = []
    
    # 2. Evaluate across splits
    # test_ua: Unseen Answers (same questions)
    # test_uq: Unseen Questions (generalization)
    # test_ud: Unseen Domains (robustness)
    for split in ["test_ua", "test_uq", "test_ud"]:
        print(f"Evaluating on {split}...")
        _, X_test, _, y_test, _, test_ref = prepare_dataframe(dataset, split)
        
        acc, f1, report = clf.evaluate(X_test, y_test, references=test_ref)
        
        results.append({
            "split": split,
            "accuracy": acc,
            "f1_macro": f1
        })
        
    df_results = pd.DataFrame(results)
    print("\nPhase 1 Results Summary:")
    print(df_results.to_string(index=False))
    
    # Save results
    df_results.to_csv("phase1/evaluation/ablation_results.csv", index=False)
    print("\nAblation results saved to phase1/evaluation/ablation_results.csv")

if __name__ == "__main__":
    run_phase1_ablation()
