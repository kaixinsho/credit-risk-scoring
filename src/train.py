from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "application_train.csv"
MODEL_PATH = ROOT / "models" / "lgbm_v1.joblib"


def add_features(df):
    df = df.copy()
    # 365243 adalah penanda "tidak bekerja / pensiun", bukan jumlah hari asli
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    return df


def prepare_X(df, categories=None):
    X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    if categories is None:
        cat_cols = X.select_dtypes("object").columns
        categories = {c: sorted(X[c].dropna().unique()) for c in cat_cols}
    for col, cats in categories.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return X, categories


def main():
    df = add_features(pd.read_csv(DATA_PATH))
    y = df["TARGET"]
    X, categories = prepare_X(df)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50)],
    )

    auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    print(f"AUC validasi: {auc:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "categories": categories,
         "features": list(X.columns), "auc": auc},
        MODEL_PATH,
    )
    print("Model disimpan di", MODEL_PATH)


if __name__ == "__main__":
    main()