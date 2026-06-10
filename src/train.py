"""
Train the PD model.

LightGBM (a gradient-boosting machine) is the deliberate choice: it's a standard,
strong credit-risk model, handles the heavy missingness in this data natively, and
gives the SHAP values the reason-code layer needs. Run:

    python -m src.train

Outputs:
    artifacts/model_fold{k}.lgb     one model per CV fold
    artifacts/oof.npy               out-of-fold predictions (honest, un-leaked)
    artifacts/oof_labels.npy        matching labels
    artifacts/feature_names.txt
"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

import config as C
from src.features import build_features
from src.metrics import auc, gini, ks_statistic

PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.02,
    num_leaves=48,
    min_child_samples=80,
    feature_fraction=0.5,        # heavy column subsampling — many correlated features
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    n_jobs=-1,
    seed=C.SEED,
    verbosity=-1,
)


def main():
    print("building features ...")
    # application_train already contains TARGET, so build_features("train") carries it.
    df = build_features("train")
    if C.TARGET not in df.columns:
        raise RuntimeError(
            f"'{C.TARGET}' not found in features. Make sure application_train.csv "
            f"(which has the TARGET column) is the one under data/homecredit/."
        )

    feature_names = [c for c in df.columns if c not in [C.ID_COL, C.TARGET]]
    X = df[feature_names].values.astype("float32")
    y = df[C.TARGET].values
    (C.ARTIFACTS / "feature_names.txt").write_text("\n".join(feature_names))
    print(f"matrix: {X.shape}, default rate: {y.mean():.4f}")

    oof = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        dtrain = lgb.Dataset(X[tr], y[tr])
        dvalid = lgb.Dataset(X[va], y[va])
        model = lgb.train(
            PARAMS, dtrain, num_boost_round=5000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
        )
        oof[va] = model.predict(X[va])
        model.save_model(str(C.ARTIFACTS / f"model_fold{k}.lgb"))
        print(f"fold {k}: AUC={auc(y[va], oof[va]):.4f}")

    np.save(C.ARTIFACTS / "oof.npy", oof)
    np.save(C.ARTIFACTS / "oof_labels.npy", y)

    print("\n=== OUT-OF-FOLD (report these, not per-fold train numbers) ===")
    print(f"AUC  : {auc(y, oof):.4f}")
    print(f"Gini : {gini(y, oof):.4f}")
    print(f"KS   : {ks_statistic(y, oof):.4f}")


if __name__ == "__main__":
    main()
