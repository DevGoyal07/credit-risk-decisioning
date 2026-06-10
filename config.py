"""
Central config. Edit DATA_DIR to wherever you unzipped the Home Credit data, then
everything downstream follows.

Dataset: Home Credit Default Risk (Kaggle). Multi-table, NAMED features -> meaningful
reason codes and visible relational feature engineering. Competition metric: AUC.
"""
from pathlib import Path

# --- Paths (put the Kaggle CSVs under data/homecredit/) ---
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data" / "homecredit"
ARTIFACTS  = ROOT / "artifacts"
REPORTS    = ROOT / "reports"
ARTIFACTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# --- Home Credit table files (the relational layers) ---
ID_COL     = "SK_ID_CURR"          # the applicant key everything joins on
TARGET     = "TARGET"              # 1 = client had payment difficulties (default)
PREV_ID    = "SK_ID_PREV"

APPLICATION_TRAIN = "application_train.csv"
APPLICATION_TEST  = "application_test.csv"
BUREAU            = "bureau.csv"
PREVIOUS          = "previous_application.csv"
INSTALLMENTS      = "installments_payments.csv"
POS_CASH          = "POS_CASH_balance.csv"
CREDIT_CARD       = "credit_card_balance.csv"

# --- Modelling ---
N_FOLDS   = 5
SEED      = 42

# --- Decisioning / policy layer ---
# Expected loss = PD * LGD * EAD. These are illustrative assumptions you must state
# explicitly when you defend the project (they are NOT measured from data).
LGD = 0.80                          # loss given default (fraction of exposure lost)
EAD = 1.0                           # exposure at default, normalized to 1 unit of limit
MARGIN = 0.10                       # expected revenue per approved good account (illustrative)

# Decision bands on calibrated PD. Tune against the approval/loss curve in policy.py.
APPROVE_BELOW = 0.08                # PD < 8%  -> approve
DECLINE_ABOVE = 0.25                # PD > 25% -> decline; in-between -> manual review ("refer")

# Number of adverse-action reason codes to surface per declined applicant
N_REASON_CODES = 4
