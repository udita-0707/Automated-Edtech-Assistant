"""
Quick alpha calibration script (runs in ~2 minutes, no notebook required).
Sweeps alpha in [0.0, 1.0] on test_ua and prints optimal value.
Usage: python phase3/scripts/calibrate_alpha.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PH3  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PH3)
_PH2  = os.path.join(_ROOT, "phase2")
for _p in [_ROOT, _PH2, _PH3]:
    if _p not in sys.path: sys.path.insert(0, _p)
    else: sys.path.remove(_p); sys.path.insert(0, _p)
if sys.path[0] != _PH3: sys.path.insert(0, _PH3)

import numpy as np
import importlib.util as _ilu
from sklearn.metrics import f1_score

import config
import utils as p3_utils
from grading.calibrated_scorer import CalibratedScorer
from grading.hybrid_grader import SynergisticHybridGrader

# Load dataset
print("[1/4] Loading dataset...")
dataset = p3_utils.load_scientsbank()
dataset = p3_utils.convert_labels(dataset, config.LABEL_SCHEME)
_, X_val, _, y_val, _, val_ref = p3_utils.prepare_dataframe(dataset, "test_ua")
X_val_list, val_ref_list, y_val_list = X_val.tolist(), val_ref.tolist(), y_val.tolist()
print(f"      Validation split: {len(X_val_list)} samples")

# Load SVM
print("[2/4] Loading ClassicalGrader...")
_spec = _ilu.spec_from_file_location("classical_grader", os.path.join(_PH2, "grading", "classical_grader.py"))
_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
cg = _mod.ClassicalGrader(); cg.load(config.MODEL_A_PATH)
print("      SVM loaded.")

# Load calibrated scorer
print("[3/4] Loading CalibratedScorer (pre-calibrated)...")
cs = CalibratedScorer()
if not cs.load_thresholds():
    print("      ERROR: run phase3/run_pipeline.py first to calibrate thresholds.")
    sys.exit(1)
print(f"      Thresholds: {cs.thresholds}")

# Alpha sweep
print("[4/4] Running alpha sweep [0.0 → 1.0] in steps of 0.05...")
alphas, f1s = [], []
for alpha in np.arange(0.0, 1.01, 0.05):
    hg = SynergisticHybridGrader(
        classical_grader=cg,
        calibrated_scorer=cs,
        alpha=float(alpha),
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )
    preds = [hg.grade(s, r)["label_idx"] for s, r in zip(X_val_list, val_ref_list)]
    f1    = f1_score(y_val_list, preds, average="macro", zero_division=0)
    alphas.append(round(float(alpha), 2))
    f1s.append(round(f1, 4))
    print(f"  alpha={alpha:.2f}  F1={f1:.4f}")

opt_alpha = alphas[int(np.argmax(f1s))]
opt_f1    = max(f1s)

print(f"\n{'='*45}")
print(f"Optimal alpha = {opt_alpha}   (F1={opt_f1:.4f})")
print(f"SVM weight:   {opt_alpha*100:.0f}%")
print(f"SBERT weight: {(1-opt_alpha)*100:.0f}%")
print(f"{'='*45}")
print(f"\nUpdate config.py: DEFAULT_ALPHA = {opt_alpha}")
