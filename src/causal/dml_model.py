# src/causal/dml_model.py
"""
dml_model.py - Tasks 2.5 / 2.6: Double ML training and validation.

Fits on TRAIN only, evaluates on TEST against the synthetic true ITE.
Run from the project root:   python src/causal/dml_model.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.dml import LinearDML, CausalForestDML

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
DOCS = PROJECT_ROOT / "docs"

TREATMENT, OUTCOME = "discount_amount", "purchased"


def load_split(split):
    X = pd.read_parquet(PROCESSED / f"X_{split}.parquet")
    meta = pd.read_parquet(PROCESSED / f"{split}_metadata.parquet")
    assert (X["row_id"].to_numpy() == meta["row_id"].to_numpy()).all(), "row_id mismatch"
    features = [c for c in X.columns if c != "row_id"]
    return X[features], meta, features


def base_regressor():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                         min_child_samples=50, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=42, verbose=-1)


def base_classifier():
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=42, verbose=-1)


def train_dml(X, Y, T, kind):
    if kind == "linear":
        est = LinearDML(model_y=base_regressor(), model_t=base_classifier(),
                        discrete_treatment=True, cv=5, random_state=42)
    elif kind == "forest":
        est = CausalForestDML(model_y=base_regressor(), model_t=base_classifier(),
                              discrete_treatment=True, cv=3, n_estimators=300,
                              min_samples_leaf=30, random_state=42)
    else:
        raise ValueError(kind)
    est.fit(Y, T, X=X)       # numpy arrays: avoids LightGBM feature-name issues
    return est


def predict_ite(est, X):
    """ITE = change in purchase probability vs $0, for $10 and $20."""
    return pd.DataFrame({
        "ite_10": est.effect(X, T0=0, T1=10),
        "ite_20": est.effect(X, T0=0, T1=20),
    })


def validate(ite, meta, name):
    T, Y = meta[TREATMENT], meta[OUTCOME]
    rows = []
    for k in (10, 20):
        true = meta[f"true_ite_{k}"].to_numpy()
        pred = ite[f"ite_{k}"].to_numpy()
        rows.append({
            "model": name, "discount": k,
            "naive_ATE": Y[T == k].mean() - Y[T == 0].mean(),
            "true_ATE": true.mean(),
            "dml_ATE": pred.mean(),
            "ITE_corr": np.corrcoef(pred, true)[0, 1],
            "ITE_RMSE": np.sqrt(np.mean((pred - true) ** 2)),
        })
    return pd.DataFrame(rows)


def quintile_check(ite, meta, name, k=20):
    """Customers ranked by predicted ITE should show a falling TRUE ITE."""
    q = pd.qcut(ite[f"ite_{k}"].rank(method="first"), 5,
                labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    out = meta[f"true_ite_{k}"].groupby(q.to_numpy(), observed=True).mean()
    print(f"\n{name}: mean TRUE ITE for ${k}, by predicted-ITE quintile")
    print(out.round(4).to_string())


def main():
    X_tr, m_tr, features = load_split("train")
    X_te, m_te, _ = load_split("test")
    Y_tr = m_tr[OUTCOME].to_numpy()
    T_tr = m_tr[TREATMENT].astype(int).to_numpy()
    X_tr_np, X_te_np = X_tr.to_numpy(float), X_te.to_numpy(float)

    MODELS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    tables = []

    for kind in ["linear", "forest"]:
        print(f"\nTraining {kind} DML ...")
        est = train_dml(X_tr_np, Y_tr, T_tr, kind)
        joblib.dump(est, MODELS / f"dml_{kind}.joblib")

        ite = predict_ite(est, X_te_np)
        tables.append(validate(ite, m_te, f"{kind}_dml"))
        quintile_check(ite, m_te, f"{kind}_dml")

        out = ite.copy()
        out.insert(0, "row_id", m_te["row_id"].to_numpy())
        out.to_parquet(PROCESSED / f"ite_test_{kind}.parquet", index=False)

    result = pd.concat(tables).round(4)
    result.to_csv(DOCS / "dml_validation.csv", index=False)
    print("\n--- VALIDATION ON TEST SET ---")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()