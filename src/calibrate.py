"""
Calibration — the step that turns a RANKING into a real probability of default (PD).

A GBM score of 0.30 does not mean "30% chance of default" out of the box. For a credit
DECISION (expected loss = PD * LGD * EAD) the number has to mean what it says, so we fit
an isotonic regression on out-of-fold predictions and check the calibration curve.

Run AFTER train.py:
    python -m src.calibrate

Outputs:
    artifacts/calibrator.pkl
    reports/calibration.png
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

import config as C


def main():
    oof = np.load(C.ARTIFACTS / "oof.npy")
    y = np.load(C.ARTIFACTS / "oof_labels.npy")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y)
    calibrated = iso.predict(oof)
    with open(C.ARTIFACTS / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)

    # Reliability curve: before vs after
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, p in [("raw GBM score", oof), ("calibrated PD", calibrated)]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed default rate")
    ax.set_title("Calibration (reliability) curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.REPORTS / "calibration.png", dpi=130)
    print("saved reports/calibration.png and artifacts/calibrator.pkl")


if __name__ == "__main__":
    main()
