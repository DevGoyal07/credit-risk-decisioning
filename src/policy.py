"""
Decisioning layer — this is what makes the project "credit-risk DECISIONING" and not
just "default prediction." It converts a PD into a business action.

Two views:
  1) decide(pd): approve / refer / decline using the bands in config (simple, explainable).
  2) approval_loss_curve(): sweep the approval threshold and plot approval rate vs
     bad-rate among approved — the trade-off a credit policy team actually argues over.
     It also finds the expected-profit-maximizing cutoff given the cost assumptions.

Run for the plot:
    python -m src.policy
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def decide(pd_value: float) -> str:
    if pd_value < C.APPROVE_BELOW:
        return "APPROVE"
    if pd_value > C.DECLINE_ABOVE:
        return "DECLINE"
    return "REFER"          # manual review band


def expected_profit(pd_value: float) -> float:
    """Per-account expected profit if approved. Approve only when this is positive."""
    exp_loss = pd_value * C.LGD * C.EAD
    exp_revenue = (1 - pd_value) * C.MARGIN
    return exp_revenue - exp_loss


def approval_loss_curve(pd_scores: np.ndarray, y_true: np.ndarray):
    """Sweep thresholds; return arrays for plotting + the profit-optimal cutoff."""
    thresholds = np.linspace(0.01, 0.6, 120)
    approval_rate, bad_rate, profit = [], [], []
    for t in thresholds:
        approved = pd_scores <= t
        approval_rate.append(approved.mean())
        bad_rate.append(y_true[approved].mean() if approved.any() else 0.0)
        # portfolio expected profit at this cutoff
        ep = np.where(approved, (1 - pd_scores) * C.MARGIN - pd_scores * C.LGD * C.EAD, 0.0)
        profit.append(ep.sum())
    profit = np.array(profit)
    best_t = thresholds[int(profit.argmax())]
    return thresholds, np.array(approval_rate), np.array(bad_rate), profit, best_t


def main():
    import pickle
    oof = np.load(C.ARTIFACTS / "oof.npy")
    y = np.load(C.ARTIFACTS / "oof_labels.npy")
    with open(C.ARTIFACTS / "calibrator.pkl", "rb") as f:
        iso = pickle.load(f)
    pd_scores = iso.predict(oof)

    t, appr, bad, profit, best_t = approval_loss_curve(pd_scores, y)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(appr, bad, color="crimson")
    ax1.set_xlabel("approval rate")
    ax1.set_ylabel("bad rate among approved", color="crimson")
    ax1.set_title("Approval vs bad-rate trade-off")
    ax1.axvline(appr[int(profit.argmax())], ls="--", color="gray")
    fig.tight_layout()
    fig.savefig(C.REPORTS / "approval_curve.png", dpi=130)
    print(f"profit-optimal PD cutoff ≈ {best_t:.3f}  "
          f"(approval {appr[int(profit.argmax())]:.1%}, "
          f"bad-rate {bad[int(profit.argmax())]:.2%})")
    print("saved reports/approval_curve.png")
    print("NOTE: the cutoff depends on LGD/EAD/MARGIN in config — they are assumptions, "
          "state them when you defend this.")


if __name__ == "__main__":
    main()
