"""
Evaluation metrics for credit scoring.

Home Credit's competition metric is AUC. Gini (= 2*AUC - 1) and KS are the standard
credit-scorecard metrics you should also be ready to explain in an interview:
  - AUC  : probability the model ranks a random defaulter above a random non-defaulter.
  - Gini : a 0-1 rescaling of AUC (0 = random, 1 = perfect ranking).
  - KS   : max gap between the cumulative good and bad distributions; scorecard teams
           quote it constantly as a single "separation" number.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, y_pred) -> float:
    return roc_auc_score(y_true, y_pred)


def gini(y_true, y_pred) -> float:
    return 2.0 * roc_auc_score(y_true, y_pred) - 1.0


def ks_statistic(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    order = np.argsort(y_pred)[::-1]
    y = y_true[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))
