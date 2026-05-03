"""
Phase 3 — End-to-End Pipeline Runner
======================================
Orchestrates: calibration → ablation → bias → explainability → summary.

Run from repository root:
    python phase3/run_pipeline.py

Prerequisites:
    1. Phase 2 artefacts must exist:
       phase2/data/artifacts/model_a.pkl
       If missing: run  python phase2/run_train_eval.py

    2. Python dependencies installed:
       pip install -r phase3/requirements.txt
       python -m spacy download en_core_web_sm
"""

import os
import sys
import logging
import json

# ---------------------------------------------------------------------------
# Path setup — supports running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PHASE2 = os.path.join(_ROOT, "phase2")

# Insert in reverse priority order (last insert → position 0 → highest priority).
# Priority: phase3 > repo_root > phase2
# This ensures phase3/config.py always wins over phase2/config.py on import.
for _p in [_ROOT, _PHASE2, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
    else:
        # Already present — re-promote to front if not phase3
        sys.path.remove(_p)
        sys.path.insert(0, _p)
# Final guarantee: phase3 directory is always sys.path[0]
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config
import utils as p3_utils

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(config.PHASE3_EVALUATION, exist_ok=True)
os.makedirs(config.PHASE3_ARTIFACTS, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.PHASE3_EVALUATION, "pipeline.log")),
    ],
)
logger = logging.getLogger("phase3.pipeline")


# ===========================================================================
# Step 0 — Validate Phase 2 artefacts
# ===========================================================================

def _require_phase2_artifacts():
    if not os.path.exists(config.MODEL_A_PATH):
        print(
            "\n" + "!" * 60 +
            f"\nERROR: Phase 2 SVM artefact not found at:\n  {config.MODEL_A_PATH}\n"
            "This file is required before running Phase 3.\n"
            "Fix: run the following command from repo root:\n"
            "     python phase2/run_train_eval.py\n" +
            "!" * 60 + "\n"
        )
        sys.exit(1)
    logger.info(f"✓ Phase 2 artefact verified: {config.MODEL_A_PATH}")


# ===========================================================================
# Step 1 — Load dataset
# ===========================================================================

def _load_dataset():
    logger.info("\n[Step 1] Loading SciEntsBank dataset...")
    dataset = p3_utils.load_scientsbank()
    dataset = p3_utils.convert_labels(dataset, config.LABEL_SCHEME)
    logger.info("✓ Dataset loaded and labels converted to 3-way scheme.")
    return dataset


# ===========================================================================
# Step 2 — Load ClassicalGrader (Phase 2 SVM)
# ===========================================================================

def _load_classical_grader():
    logger.info("\n[Step 2] Loading Phase 2 ClassicalGrader (SVM)...")
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "classical_grader",
        os.path.join(_PHASE2, "grading", "classical_grader.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ClassicalGrader = _mod.ClassicalGrader
    cg = ClassicalGrader()
    cg.load(config.MODEL_A_PATH)
    logger.info("✓ ClassicalGrader loaded.")
    return cg


# ===========================================================================
# Step 3 — Calibrate SBERT thresholds
# ===========================================================================

def _calibrate_sbert(dataset):
    logger.info("\n[Step 3] Calibrating SBERT thresholds on training split...")
    from grading.calibrated_scorer import CalibratedScorer

    cs = CalibratedScorer(model_name=config.SBERT_MODEL)

    # Get training data
    X_train, _, y_train, _, train_ref, _ = p3_utils.prepare_dataframe(dataset, "train")

    thresholds = cs.calibrate(X_train, train_ref, y_train, force=False)
    logger.info(f"✓ Calibration complete. Thresholds: {thresholds}")
    return cs


# ===========================================================================
# Step 4 — Build Synergistic Hybrid Grader
# ===========================================================================

def _build_hybrid(classical_grader, calibrated_scorer):
    logger.info("\n[Step 4] Assembling SynergisticHybridGrader...")
    from grading.hybrid_grader import SynergisticHybridGrader

    hybrid = SynergisticHybridGrader(
        classical_grader=classical_grader,
        calibrated_scorer=calibrated_scorer,
        alpha=config.DEFAULT_ALPHA,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )
    logger.info(
        f"✓ SynergisticHybridGrader assembled  "
        f"(α={config.DEFAULT_ALPHA}, conf_threshold={config.CONFIDENCE_THRESHOLD})"
    )
    return hybrid


# ===========================================================================
# Step 5 — Run full ablation study
# ===========================================================================

def _run_ablation(dataset, classical_grader, calibrated_scorer, hybrid_grader):
    logger.info("\n[Step 5] Running Phase 3 ablation study...")
    from evaluation.ablation import run_ablation

    df = run_ablation(
        dataset=dataset,
        classical_grader=classical_grader,
        calibrated_scorer=calibrated_scorer,
        hybrid_grader=hybrid_grader,
    )
    logger.info(f"✓ Ablation complete. CSV → {config.ABLATION_CSV_PATH}")
    return df


# ===========================================================================
# Step 6 — Bias analysis
# ===========================================================================

def _run_bias_analysis(dataset, hybrid_grader, calibrated_scorer):
    logger.info("\n[Step 6] Running bias analysis on test_ua with Model C...")
    from evaluation.bias_analysis import (
        analyze_length_bias, analyze_domain_bias,
        generate_bias_report, plot_bias_summary,
    )

    _, X_test, _, y_test, _, test_ref = p3_utils.prepare_dataframe(dataset, "test_ua")
    X_list   = X_test.tolist()
    ref_list = test_ref.tolist()
    y_true   = y_test.tolist()

    hybrid_grader.reset_counters()
    y_pred = [hybrid_grader.grade(s, r)["label_idx"] for s, r in zip(X_list, ref_list)]

    # Length bias
    length_bias = analyze_length_bias(X_list, y_true, y_pred)
    logger.info(
        f"  Length bias: detected={length_bias['bias_detected']}, "
        f"magnitude={length_bias['bias_magnitude']:.4f}"
    )

    # Domain bias — collect F1 per split for Model C
    results_by_split = {}
    for split in ["test_ua", "test_uq", "test_ud"]:
        from sklearn.metrics import f1_score as _f1
        _, Xt, _, yt, _, rt = p3_utils.prepare_dataframe(dataset, split)
        yp = [hybrid_grader.grade(s, r)["label_idx"]
              for s, r in zip(Xt.tolist(), rt.tolist())]
        results_by_split[split] = {
            "f1_macro": _f1(yt.tolist(), yp, average="macro", zero_division=0)
        }

    domain_bias = analyze_domain_bias(results_by_split)
    logger.info(
        f"  Domain gap (UA→UD): {domain_bias['domain_gap']:.4f}, "
        f"detected={domain_bias['domain_bias_detected']}"
    )

    # Save report
    report_str = generate_bias_report(length_bias, domain_bias)
    with open(config.BIAS_REPORT_PATH, "w") as fh:
        fh.write(report_str)
    logger.info(f"✓ Bias report saved → {config.BIAS_REPORT_PATH}")

    # Model results dict for plot
    model_results_plot = {
        "Model_C_Synergistic": {sp: v for sp, v in results_by_split.items()}
    }

    plot_bias_summary(
        length_bias, domain_bias,
        model_results_plot,
        config.PHASE3_EVALUATION,
    )

    return length_bias, domain_bias


# ===========================================================================
# Step 7 — SHAP explainability
# ===========================================================================

def _run_shap(dataset, classical_grader):
    logger.info("\n[Step 7] Running SHAP explainability...")
    from evaluation.explainability import explain_svm_predictions

    _, X_test, _, _, _, test_ref = p3_utils.prepare_dataframe(dataset, "test_ua")

    top_features = explain_svm_predictions(
        classical_grader=classical_grader,
        X_test=X_test.tolist(),
        references=test_ref.tolist(),
        save_path=config.PHASE3_EVALUATION,
        n_samples=200,
    )

    if top_features is not None and len(top_features) > 0:
        logger.info(f"✓ SHAP complete. Top-10 features:\n{top_features[['feature','mean_abs_shap']].to_string(index=False)}")
    else:
        logger.warning("SHAP returned empty results — check shap/sklearn versions.")

    return top_features


# ===========================================================================
# Step 8 — Final summary
# ===========================================================================

def _print_final_summary(
    ablation_df,
    length_bias,
    domain_bias,
    hybrid_grader,
    calibrated_scorer,
):
    import pandas as pd

    def _get_f1(model, split):
        row = ablation_df[
            (ablation_df["Model"] == model) &
            (ablation_df["Split"] == split)
        ]
        return float(row["F1_Macro"].iloc[0]) if len(row) > 0 else 0.0

    ua_f1 = _get_f1("Model_C_Synergistic", "test_ua")
    uq_f1 = _get_f1("Model_C_Synergistic", "test_uq")
    ud_f1 = _get_f1("Model_C_Synergistic", "test_ud")

    p1_ua  = 0.5732
    p2_ua  = 0.5294   # Phase 2 best SVM (alpha=1.0)
    improvement = ua_f1 - p1_ua

    path_stats = hybrid_grader.path_stats()

    banner = (
        "\n" + "═" * 50 + "\n"
        "Phase 3 Pipeline Complete\n" +
        "═" * 50 + "\n\n"
        f"Calibrated SBERT thresholds:\n"
        f"  correct  ≥ {calibrated_scorer.thresholds['correct']}\n"
        f"  partial  ≥ {calibrated_scorer.thresholds['partial']}\n\n"
        "Best model: Synergistic Hybrid (Model C)\n"
        f"  test_ua F1: {ua_f1:.4f}\n"
        f"  test_uq F1: {uq_f1:.4f}\n"
        f"  test_ud F1: {ud_f1:.4f}\n\n"
        f"vs Phase 1 baseline (test_ua): {p1_ua:.4f}\n"
        f"vs Phase 2 best SVM (test_ua): {p2_ua:.4f}\n"
        f"Improvement over Phase 1:      {improvement:+.4f}\n\n"
        f"Fast path used: {path_stats['fast_pct']:.1f}% of predictions\n"
        f"Domain gap (UA→UD): {domain_bias['domain_gap']:.4f}\n"
        f"Length bias detected: {length_bias['bias_detected']}\n\n"
        f"Artifacts saved to: {config.PHASE3_ARTIFACTS}\n"
        f"SHAP plot:          {os.path.join(config.PHASE3_EVALUATION, 'shap_summary.png')}\n"
        f"Ablation CSV:       {config.ABLATION_CSV_PATH}\n"
        f"Bias report:        {config.BIAS_REPORT_PATH}\n" +
        "═" * 50
    )
    print(banner)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "═" * 50)
    print("Phase 3 — Automated EdTech Grading Assistant")
    print("Synergistic Hybrid Pipeline")
    print("═" * 50 + "\n")

    _require_phase2_artifacts()

    dataset           = _load_dataset()
    classical_grader  = _load_classical_grader()
    calibrated_scorer = _calibrate_sbert(dataset)
    hybrid_grader     = _build_hybrid(classical_grader, calibrated_scorer)

    ablation_df       = _run_ablation(dataset, classical_grader,
                                      calibrated_scorer, hybrid_grader)
    length_bias, domain_bias = _run_bias_analysis(dataset, hybrid_grader,
                                                   calibrated_scorer)
    _run_shap(dataset, classical_grader)

    _print_final_summary(
        ablation_df, length_bias, domain_bias,
        hybrid_grader, calibrated_scorer,
    )


if __name__ == "__main__":
    main()
