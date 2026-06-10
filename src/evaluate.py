"""
Evaluation harness — produces the numbers you (only later, once real) put on the resume.

Run AFTER train.py and calibrate.py:
    python -m src.evaluate

Writes reports/results.md with AUC, Gini, KS, and a Brier score (calibration quality).
Do NOT copy these onto your resume until this has actually run on the real data.
"""
import pickle
import numpy as np
from sklearn.metrics import brier_score_loss

import config as C
from src.metrics import auc, gini, ks_statistic


def main():
    oof = np.load(C.ARTIFACTS / "oof.npy")
    y = np.load(C.ARTIFACTS / "oof_labels.npy")
    with open(C.ARTIFACTS / "calibrator.pkl", "rb") as f:
        iso = pickle.load(f)
    pd_scores = iso.predict(oof)

    rows = {
        "AUC":                auc(y, oof),
        "Gini":               gini(y, oof),
        "KS":                 ks_statistic(y, oof),
        "Brier (calibrated)": brier_score_loss(y, pd_scores),
        "Default rate (base)": float(y.mean()),
        "N applicants":       int(len(y)),
    }

    md = ["# Results\n", "_Out-of-fold, measured on real data._\n",
          "| Metric | Value |", "|---|---|"]
    for k, v in rows.items():
        md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v:,} |")
    md += ["", "![Calibration](calibration.png)", "",
           "![Approval curve](approval_curve.png)", ""]
    (C.REPORTS / "results.md").write_text("\n".join(md))

    for k, v in rows.items():
        print(f"{k:>22}: {v:.4f}" if isinstance(v, float) else f"{k:>22}: {v:,}")
    print("\nwrote reports/results.md")


if __name__ == "__main__":
    main()
