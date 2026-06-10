"""
Feature engineering — the relational aggregation work interviewers will push on hardest.

Home Credit is MULTI-TABLE: one application row per client, plus several one-to-many
history tables (prior bureau credits, prior Home Credit applications, installment
payments, POS/cash balances, credit-card balances). A GBM needs one row per client, so
each history table is aggregated to SK_ID_CURR and joined onto the application.

The aggregations encode real credit intuition:
  - application ratios: credit/income, annuity/income, employment/age  -> affordability
  - bureau: count + share of ACTIVE prior credits, total debt, overdue  -> external load
  - previous: approval/refusal rates on prior HC applications           -> relationship
  - installments: late days (DPD) and under-payment                     -> repayment behaviour
  - POS / credit card: days-past-due and utilization                    -> current stress

TODO (your edges, in rough order of payoff):
  1. Recency-weight the installment aggregations (last 12 months matter most).
  2. Add bureau_balance STATUS history (months in DPD buckets).
  3. EXT_SOURCE interactions (EXT_SOURCE_1*2*3, mean, std) — famously the strongest signals.
"""
import numpy as np
import pandas as pd

import config as C
from src.data import load_application, load_table


def _agg_numeric(df, group_col, num_cols, prefix):
    """mean/max/min/sum/std of numeric cols, grouped, with a record count."""
    g = df.groupby(group_col)
    agg = g[num_cols].agg(["mean", "max", "min", "sum", "std"])
    agg.columns = [f"{prefix}_{c}_{stat}".upper() for c, stat in agg.columns]
    agg[f"{prefix}_COUNT"] = g.size()
    return agg.reset_index()


def application_features(split="train") -> pd.DataFrame:
    df = load_application(split).copy()

    # Known data-quality fix: DAYS_EMPLOYED uses 365243 as a "not employed" sentinel.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Affordability / leverage ratios (all defensible, all named)
    df["RATIO_CREDIT_INCOME"]   = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["RATIO_ANNUITY_INCOME"]  = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["RATIO_CREDIT_ANNUITY"]  = df["AMT_CREDIT"] / df["AMT_ANNUITY"]          # ~ loan term
    df["RATIO_EMPLOYED_AGE"]    = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["EXT_SOURCE_MEAN"]       = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_STD"]        = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)

    # One-hot the categoricals (mostly low-cardinality). Numeric-only matrix downstream.
    df = pd.get_dummies(df, dummy_na=True)
    return df


def bureau_features() -> pd.DataFrame:
    b = load_table(C.BUREAU)
    num = ["DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
           "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
           "AMT_CREDIT_SUM_LIMIT", "CNT_CREDIT_PROLONG"]
    num = [c for c in num if c in b.columns]
    feats = _agg_numeric(b, C.ID_COL, num, "BURO")
    # share of currently-active external credits
    active = (b.assign(_active=(b["CREDIT_ACTIVE"] == "Active").astype(int))
                .groupby(C.ID_COL)["_active"].mean().rename("BURO_ACTIVE_SHARE"))
    return feats.merge(active.reset_index(), on=C.ID_COL, how="left")


def previous_features() -> pd.DataFrame:
    p = load_table(C.PREVIOUS)
    num = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
           "DAYS_DECISION", "CNT_PAYMENT"]
    num = [c for c in num if c in p.columns]
    feats = _agg_numeric(p, C.ID_COL, num, "PREV")
    refused = (p.assign(_ref=(p["NAME_CONTRACT_STATUS"] == "Refused").astype(int))
                 .groupby(C.ID_COL)["_ref"].mean().rename("PREV_REFUSED_SHARE"))
    return feats.merge(refused.reset_index(), on=C.ID_COL, how="left")


def installment_features() -> pd.DataFrame:
    ins = load_table(C.INSTALLMENTS)
    # repayment behaviour: positive DPD = paid late; positive DIFF = paid less than due
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)
    return _agg_numeric(ins, C.ID_COL, ["DPD", "PAYMENT_DIFF", "PAYMENT_RATIO"], "INS")


def pos_features() -> pd.DataFrame:
    pos = load_table(C.POS_CASH)
    num = [c for c in ["SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]
           if c in pos.columns]
    return _agg_numeric(pos, C.ID_COL, num, "POS")


def credit_card_features() -> pd.DataFrame:
    cc = load_table(C.CREDIT_CARD)
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    num = [c for c in ["UTILIZATION", "AMT_BALANCE", "SK_DPD", "AMT_DRAWINGS_CURRENT"]
           if c in cc.columns]
    return _agg_numeric(cc, C.ID_COL, num, "CC")


def build_features(split="train") -> pd.DataFrame:
    """Join every aggregated history table onto the application table by SK_ID_CURR."""
    feats = application_features(split)
    for fn in (bureau_features, previous_features, installment_features,
               pos_features, credit_card_features):
        try:
            feats = feats.merge(fn(), on=C.ID_COL, how="left")
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    return feats


if __name__ == "__main__":
    f = build_features("train")
    print("feature matrix:", f.shape)
    print(f.filter(like="RATIO_").head(3))
