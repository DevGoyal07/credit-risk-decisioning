"""
Adverse-action reason codes — the feature that signals you understand credit, not just ML.

US lenders must tell a declined applicant WHY (ECOA / Reg B "adverse action notice").
We approximate that: SHAP attributes an applicant's PD to individual features; the
features pushing risk UP the most become the reason codes.

Home Credit features are NAMED, so the reasons read naturally. FEATURE_GLOSSARY maps
the raw engineered column to a customer-facing sentence — extend it as you add features.
Use HomeCredit_columns_description.csv (the data dictionary) to look up any feature you
don't recognise, then add a line here.
"""
import numpy as np
import lightgbm as lgb
import shap

import config as C

# Engineered column -> plain-language adverse-action reason. Extend freely.
FEATURE_GLOSSARY = {
    # --- external bureau scores (usually the strongest signals) ---
    "EXT_SOURCE_MEAN":        "Low external credit-bureau scores",
    "EXT_SOURCE_1":           "Low external credit score (source 1)",
    "EXT_SOURCE_2":           "Low external credit score (source 2)",
    "EXT_SOURCE_3":           "Low external credit score (source 3)",
    # --- affordability / leverage ---
    "RATIO_CREDIT_INCOME":    "Requested loan is large relative to income",
    "RATIO_ANNUITY_INCOME":   "Repayment burden is high relative to income",
    "RATIO_CREDIT_ANNUITY":   "Long effective loan term",
    "RATIO_EMPLOYED_AGE":     "Short employment history for age",
    "AMT_CREDIT":             "Large requested loan amount",
    "AMT_ANNUITY":            "High scheduled repayment amount",
    "AMT_GOODS_PRICE":        "High value of financed goods",
    # --- stability / demographics ---
    "DAYS_EMPLOYED":          "Short or unstable employment history",
    "DAYS_BIRTH":             "Age-related risk factor",
    "DAYS_LAST_PHONE_CHANGE": "Phone number changed recently",
    "DAYS_ID_PUBLISH":        "Identity document updated recently",
    "REGION_RATING_CLIENT":   "Applicant's region carries a higher risk rating",
    "REGION_RATING_CLIENT_W_CITY": "Applicant's region/city carries a higher risk rating",
    # --- social-circle default signals ---
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults among the applicant's social circle (30 days past due)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Defaults among the applicant's social circle (60 days past due)",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Higher-risk profile in the applicant's social circle",
    # --- aggregated history tables ---
    "BURO_ACTIVE_SHARE":      "High share of active external credits",
    "BURO_AMT_CREDIT_SUM_DEBT_SUM": "High outstanding debt at other lenders",
    "PREV_REFUSED_SHARE":     "Prior applications were frequently refused",
    "INS_DPD_MEAN":           "History of late installment payments",
    "INS_PAYMENT_DIFF_MEAN":  "History of paying less than the amount due",
    "CC_UTILIZATION_MEAN":    "High credit-card utilization",
    "POS_SK_DPD_MEAN":        "Days-past-due on prior loans",
}


def _humanize(name: str) -> str:
    # exact match first, then strip the aggregation suffix (e.g. _MEAN/_MAX/_SUM)
    if name in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[name]
    base = name.rsplit("_", 1)[0]
    return FEATURE_GLOSSARY.get(base, name)


def explain(feature_row: np.ndarray, feature_names=None, n=C.N_REASON_CODES):
    model = lgb.Booster(model_file=str(C.ARTIFACTS / "model_fold0.lgb"))
    if feature_names is None:
        feature_names = (C.ARTIFACTS / "feature_names.txt").read_text().splitlines()

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(np.asarray(feature_row, dtype=float).reshape(1, -1))
    sv = sv[1] if isinstance(sv, list) else sv      # positive (default) class
    contrib = sv.ravel()

    reasons = []
    for i in np.argsort(contrib)[::-1][:n]:
        if contrib[i] <= 0:
            continue
        name = feature_names[i]
        reasons.append((name, float(contrib[i]), _humanize(name)))
    return reasons


if __name__ == "__main__":
    print("Import `explain` and pass one applicant's feature row. See serve/score.py.")
