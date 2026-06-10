"""
Data loading for Home Credit. Step 0 of the project.

The CSVs are moderate-sized; you can read them directly. `load_table` is a thin
helper so the rest of the code never hard-codes file names.

    python -m src.data        # sanity-check shapes
"""
import pandas as pd

import config as C


def load_table(filename: str) -> pd.DataFrame:
    path = C.DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Put the Kaggle Home Credit CSVs under {C.DATA_DIR}."
        )
    return pd.read_csv(path)


def load_application(split: str = "train") -> pd.DataFrame:
    fname = C.APPLICATION_TRAIN if split == "train" else C.APPLICATION_TEST
    return load_table(fname)


def load_labels() -> pd.DataFrame:
    """SK_ID_CURR -> TARGET, taken from application_train."""
    app = load_application("train")
    return app[[C.ID_COL, C.TARGET]].copy()


if __name__ == "__main__":
    app = load_application("train")
    print("application_train:", app.shape)
    print("default rate:", round(app[C.TARGET].mean(), 4))
    for f in [C.BUREAU, C.PREVIOUS, C.INSTALLMENTS, C.POS_CASH, C.CREDIT_CARD]:
        try:
            print(f"{f}:", load_table(f).shape)
        except FileNotFoundError as e:
            print("missing:", f)
