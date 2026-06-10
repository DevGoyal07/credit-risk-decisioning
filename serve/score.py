"""
End-to-end single-applicant scorer — the demo you walk an interviewer through.

Given one applicant's engineered feature row it returns:
    - calibrated probability of default (PD)
    - the policy decision (APPROVE / REFER / DECLINE)
    - the top adverse-action reason codes (on a decline)

    python -m serve.score        # smoke test on the first training applicant
"""
import pickle
import numpy as np
import lightgbm as lgb

import config as C
from src.reason_codes import explain
from src.policy import decide, expected_profit


def score_applicant(feature_row: np.ndarray, feature_names=None) -> dict:
    feature_row = np.asarray(feature_row, dtype=float).ravel()

    raw = []
    for k in range(C.N_FOLDS):
        path = C.ARTIFACTS / f"model_fold{k}.lgb"
        if path.exists():
            raw.append(lgb.Booster(model_file=str(path)).predict(feature_row.reshape(1, -1))[0])
    raw_score = float(np.mean(raw))

    with open(C.ARTIFACTS / "calibrator.pkl", "rb") as f:
        iso = pickle.load(f)
    pd_value = float(iso.predict([raw_score])[0])

    decision = decide(pd_value)
    result = {
        "pd": round(pd_value, 4),
        "decision": decision,
        "expected_profit_if_approved": round(expected_profit(pd_value), 4),
    }
    if decision != "APPROVE":
        result["reason_codes"] = [
            {"feature": n, "impact": round(c, 4), "reason": t}
            for n, c, t in explain(feature_row, feature_names)
        ]
    return result


if __name__ == "__main__":
    try:
        from src.features import build_features
        feats = build_features("train")
        names = (C.ARTIFACTS / "feature_names.txt").read_text().splitlines()
        row = feats[names].iloc[0].values
        print(score_applicant(row, names))
    except Exception as e:
        print("Smoke test needs trained models + features. Error:", e)
