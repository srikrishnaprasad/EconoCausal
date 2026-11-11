"""
dml_model.py  -  Tasks 2.5 / 2.6 : estimate every customer's discount effect (ITE)
and validate it against the synthetic answer key.

Models compared (all fitted on TRAIN only, scored on TEST):
  1. Naive           simple purchase-rate difference            (biased, shown as a warning)
  2. T-Learner       one LightGBM per discount level            (baseline)
  3. LinearDML       Double ML, effect linear in the features
  4. CausalForestDML Double ML with a causal forest             (usually the best)

ITE_k = P(buy | $k) - P(buy | $0)  for k = 10, 20

Needs the files written by preprocess.py.
Run from anywhere:   python src/causal/dml_model.py
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import lru_cache
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.dml import LinearDML, CausalForestDML

# We only use point estimates, never LinearDML confidence intervals, so this one
# known warning (collinear one-hot columns) does not affect our results.
warnings.filterwarnings("ignore", message="Co-variance matrix is underdetermined")

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
DOCS = PROJECT_ROOT / "docs"

TREATMENT, OUTCOME = "discount_amount", "purchased"
LEVELS = [10, 20]                 # discounts compared with $0
SEED = 42
MODEL_NAMES = ["T-Learner", "LinearDML", "CausalForestDML"]
QUINTILES = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]


def regressor():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                         min_child_samples=50, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=SEED, verbose=-1)


def classifier(n_estimators=200):
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=SEED, verbose=-1)


# ------------------------------------------------------------------
# 1. LOAD AND CHECK
# ------------------------------------------------------------------
def load_split(split):
    """Return features (DataFrame), metadata (treatment, outcome, answer key), feature names."""
    X = pd.read_parquet(PROCESSED / f"X_{split}.parquet")
    meta = pd.read_parquet(PROCESSED / f"{split}_metadata.parquet")

    # feature row i must be the same customer as metadata row i
    assert (X["row_id"].to_numpy() == meta["row_id"].to_numpy()).all(), f"row_id mismatch ({split})"
    assert set(meta[TREATMENT].unique()) <= {0, 10, 20}, "unexpected discount values"
    assert set(meta[OUTCOME].unique()) <= {0, 1}, "purchased must be 0/1"

    features = [c for c in X.columns if c != "row_id"]     # the answer key is never in here
    assert not any(c.startswith("true_") for c in features), "answer key leaked into features"
    return X[features], meta, features


# ------------------------------------------------------------------
# 2. MODELS  (each predict_* returns {10: ITE array, 20: ITE array})
# ------------------------------------------------------------------
def fit_t_learner(X, T, Y):
    """One classifier per discount level, trained only on customers who got that level."""
    return {lvl: classifier().fit(X[T == lvl], Y[T == lvl]) for lvl in [0] + LEVELS}


def predict_t_learner(models, X):
    p = {lvl: models[lvl].predict_proba(X)[:, 1] for lvl in models}
    return {k: p[k] - p[0] for k in LEVELS}


def fit_dml(X, T, Y, kind):
    """Double ML: (1) predict Y from X, (2) predict T from X, (3) relate the leftovers."""
    common = dict(model_y=regressor(), model_t=classifier(),
                  discrete_treatment=True, random_state=SEED)
    if kind == "linear":
        est = LinearDML(cv=5, **common)
    else:
        est = CausalForestDML(cv=3, n_estimators=300, min_samples_leaf=30, **common)
    est.fit(Y, T, X=X)
    return est


def predict_dml(est, X):
    return {k: np.asarray(est.effect(X, T0=0, T1=k)).ravel() for k in LEVELS}


# ------------------------------------------------------------------
# 3. VALIDATION AGAINST THE ANSWER KEY
# ------------------------------------------------------------------
def validate(name, ite, meta):
    rows = []
    for k in LEVELS:
        true, pred = meta[f"true_ite_{k}"].to_numpy(), np.asarray(ite[k])
        rows.append({"model": name, "discount": f"${k}",
                     "estimated ATE": pred.mean(), "true ATE": true.mean(),
                     "ITE correlation": np.corrcoef(pred, true)[0, 1],
                     "ITE RMSE": np.sqrt(np.mean((pred - true) ** 2)),
                     "ITE MAE": np.mean(np.abs(pred - true))})
    return rows


def naive_rows(meta):
    rate = meta.groupby(TREATMENT)[OUTCOME].mean()
    return [{"model": "Naive", "discount": f"${k}",
             "estimated ATE": rate[k] - rate[0], "true ATE": meta[f"true_ite_{k}"].mean(),
             "ITE correlation": np.nan, "ITE RMSE": np.nan, "ITE MAE": np.nan} for k in LEVELS]


def quintile_table(ite, meta, k):
    """Rank customers by PREDICTED ITE, then average the TRUE ITE in each fifth.
    A good model gives a clear low-to-high staircase."""
    pred = pd.Series(np.asarray(ite[k]))
    q = pd.qcut(pred.rank(method="first"), 5, labels=QUINTILES)
    return meta[f"true_ite_{k}"].reset_index(drop=True).groupby(q, observed=True).mean()


# ------------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------------
def main():
    X_tr, m_tr, features = load_split("train")
    X_te, m_te, _ = load_split("test")
    Xtr, Xte = X_tr.to_numpy(float), X_te.to_numpy(float)   # numpy avoids LightGBM name issues
    T = m_tr[TREATMENT].astype(int).to_numpy()
    Y = m_tr[OUTCOME].to_numpy()
    print(f"Step 1  Loaded: train {Xtr.shape} | test {Xte.shape} | features {len(features)}")

    preds = {}
    t_models = fit_t_learner(Xtr, T, Y)
    preds["T-Learner"] = predict_t_learner(t_models, Xte)
    print("Step 2  T-Learner done")
    linear = fit_dml(Xtr, T, Y, "linear")
    preds["LinearDML"] = predict_dml(linear, Xte)
    print("Step 3  LinearDML done")
    forest = fit_dml(Xtr, T, Y, "forest")
    preds["CausalForestDML"] = predict_dml(forest, Xte)
    print("Step 4  CausalForestDML done")

    # ---- results table ----
    rows = naive_rows(m_te)
    for name in MODEL_NAMES:
        rows += validate(name, preds[name], m_te)
    results = pd.DataFrame(rows)
    DOCS.mkdir(exist_ok=True)
    results.round(4).to_csv(DOCS / "dml_results.csv", index=False)
    print("\nRESULTS ON TEST CUSTOMERS  (0.01 = 1 percentage point)")
    print(results.round(4).to_string(index=False))

    # ---- ranking check ----
    quint = pd.DataFrame({f"{n} ${k}": quintile_table(preds[n], m_te, k)
                          for n in MODEL_NAMES for k in LEVELS})
    quint.round(4).to_csv(DOCS / "dml_quintiles.csv")
    print("\nMEAN TRUE EFFECT inside each fifth of customers ranked by PREDICTED effect")
    print(quint[[f"{n} $20" for n in MODEL_NAMES]].round(4).to_string())

    # ---- charts ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    true20 = m_te["true_ite_20"].to_numpy() * 100
    for ax, name in zip(axes, MODEL_NAMES):
        p20 = np.asarray(preds[name][20]) * 100
        ax.scatter(true20, p20, s=3, alpha=0.2)
        lim = [min(true20.min(), p20.min()), max(true20.max(), p20.max())]
        ax.plot(lim, lim, color="red", linewidth=1)
        ax.set_title(f"{name} (correlation {np.corrcoef(true20, p20)[0, 1]:.2f})")
        ax.set_xlabel("TRUE effect of $20 discount (percentage points)")
        ax.set_ylabel("PREDICTED effect (percentage points)")
    plt.tight_layout()
    plt.savefig(DOCS / "dml_validation.png", dpi=120)
    plt.close()

    ax = (quint[[f"{n} $20" for n in MODEL_NAMES]] * 100).plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Customers ranked by PREDICTED effect of $20 (fifths)")
    ax.set_ylabel("Average TRUE effect (percentage points)")
    ax.set_title("Does the ranking work? (staircase going up = yes)")
    plt.tight_layout()
    plt.savefig(DOCS / "dml_quintiles.png", dpi=120)
    plt.close()

    # ---- save for the next steps (Qini, optimizer, API) ----
    out = m_te[["row_id", "user_id"]].copy()
    out["p0"] = t_models[0].predict_proba(Xte)[:, 1]        # chance of buying with no discount
    for name in MODEL_NAMES:
        tag = name.replace("-", "").lower()
        for k in LEVELS:
            out[f"{tag}_ite_{k}"] = np.asarray(preds[name][k])
    out.to_parquet(PROCESSED / "ite_test.parquet", index=False)

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"t_learner": t_models, "linear_dml": linear, "causal_forest": forest,
                 "features": features}, MODELS / "dml_models.joblib")
    print("\nSaved: docs/dml_results.csv, dml_quintiles.csv, dml_validation.png, dml_quintiles.png")
    print("       data/processed/ite_test.parquet, models/dml_models.joblib")

@lru_cache(maxsize=1)
def _load_bundle():
    """Load the saved models once (the file is ~50 MB), then reuse - important for the API."""
    return joblib.load(MODELS / "dml_models.joblib")

def predict_ite(df, model="causal_forest"):
    """Raw customer rows in -> user_id, p0, ite_10, ite_20 out. Used by the API."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.preprocess import transform_new_data

    bundle = _load_bundle() 
    X_new = transform_new_data(df).drop(columns="original_index")
    Xn = X_new[bundle["features"]].to_numpy(float)

    out = pd.DataFrame(index=df.index)
    if "user_id" in df.columns:
        out["user_id"] = df["user_id"].to_numpy()
    out["p0"] = bundle["t_learner"][0].predict_proba(Xn)[:, 1]
    for k in LEVELS:
        out[f"ite_{k}"] = np.asarray(bundle[model].effect(Xn, T0=0, T1=k)).ravel()
    return out.reset_index(drop=True)


if __name__ == "__main__":
    main()