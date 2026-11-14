"""services.py - glue between the API, the saved models, the optimizer and the reports."""
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data.preprocess import FEATURE_COLS, NUMERIC_FEATURES, transform_new_data
from src.optimization.allocator import recommend

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Retail_synthetic_dataset.csv"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
MODEL_FILE = ROOT / "models" / "dml_models.joblib"
trapz = getattr(np, "trapezoid", None) or np.trapz

MODEL_LABELS = {"t_learner": "T-Learner", "linear_dml": "LinearDML", "causal_forest": "CausalForestDML"}
MODEL_TAG = {"t_learner": "tlearner", "linear_dml": "lineardml", "causal_forest": "causalforestdml"}
SEGMENTS = ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]
SEG_THRESHOLD = 0.02          # effect of the $20 discount beyond +/- 2 percentage points

DATASETS = {}                 # dataset_id -> {"name", "raw", "scores": {model_key: DataFrame}}
RESULTS = {}                  # dataset_id -> last allocation (targeted customers)
_bundle = None
_reference = None


def _clean(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError("models/dml_models.joblib not found. Run src/causal/dml_model.py first.")
        _bundle = joblib.load(MODEL_FILE)
    return _bundle


def score_all(df):
    """Score every customer with all three saved models (preprocessing runs once)."""
    b = bundle()
    X = transform_new_data(df).drop(columns="original_index")
    Xn = X[b["features"]].to_numpy(float)
    p = {lvl: b["t_learner"][lvl].predict_proba(Xn)[:, 1] for lvl in (0, 10, 20)}
    base = pd.DataFrame({"user_id": df["user_id"].to_numpy(), "p0": p[0]})
    out = {}
    t = base.copy()
    for k in (10, 20):
        t[f"ite_{k}"] = p[k] - p[0]
    out["t_learner"] = t
    for key in ("linear_dml", "causal_forest"):
        d = base.copy()
        for k in (10, 20):
            d[f"ite_{k}"] = np.asarray(b[key].effect(Xn, T0=0, T1=k)).ravel()
        out[key] = d
    return out


def register(df, name):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing required columns: {missing}")
    df = df.reset_index(drop=True).copy()
    if "user_id" not in df.columns:
        df.insert(0, "user_id", np.arange(1, len(df) + 1))
    scores = score_all(df)
    dataset_id = uuid.uuid4().hex[:8]
    DATASETS[dataset_id] = {"name": name, "raw": df, "scores": scores}
    return {"dataset_id": dataset_id, "name": name, "n_rows": int(len(df)), "columns": list(df.columns),
            "checks": {"required_columns": len(FEATURE_COLS),
                       "missing_cells": int(df[FEATURE_COLS].isna().sum().sum()),
                       "has_treatment": "discount_amount" in df.columns,
                       "has_outcome": "purchased" in df.columns,
                       "has_ground_truth": "true_ite_20" in df.columns}}


def get(dataset_id):
    if dataset_id not in DATASETS:
        raise KeyError(dataset_id)
    return DATASETS[dataset_id]


def segments(sc):
    ite, p0 = sc["ite_20"].to_numpy(), sc["p0"].to_numpy()
    seg = np.where(ite > SEG_THRESHOLD, "Persuadable",
                   np.where(ite < -SEG_THRESHOLD, "Sleeping Dog",
                            np.where(p0 >= 0.5, "Sure Thing", "Lost Cause")))
    return pd.Series(seg, index=sc.index)


def summary(ds, model):
    sc, raw = ds["scores"][model], ds["raw"]
    counts = segments(sc).value_counts()
    out = {"n_customers": int(len(sc)), "model": model,
           "ate_10": float(sc["ite_10"].mean()), "ate_20": float(sc["ite_20"].mean()),
           "segments": {s: int(counts.get(s, 0)) for s in SEGMENTS}}
    if "discount_amount" in raw:
        d = raw["discount_amount"]
        out["n_treated"], out["n_control"] = int((d > 0).sum()), int((d == 0).sum())
        if "purchased" in raw:
            rate = raw.groupby("discount_amount")["purchased"].mean()
            for k in (10, 20):
                if k in rate.index and 0 in rate.index:
                    out[f"naive_ate_{k}"] = float(rate[k] - rate[0])
    for k in (10, 20):
        if f"true_ite_{k}" in raw:
            out[f"true_ate_{k}"] = float(raw[f"true_ite_{k}"].mean())
    return out


def ite_hist(sc, k, bins=40):
    h, e = np.histogram(sc[f"ite_{k}"].to_numpy(), bins=bins)
    return {"edges": e.round(5).tolist(), "counts": h.tolist()}


def predicted_curve(sc, k, points=101):
    gain = np.sort(sc[f"ite_{k}"].to_numpy())[::-1]
    n = len(gain)
    y = np.r_[0.0, np.cumsum(gain) / n]
    x = np.r_[0.0, np.arange(1, n + 1) / n]
    grid = np.linspace(0, 1, points)
    gy = np.interp(grid, x, y)
    return {"x": grid.round(4).tolist(), "model_y": gy.round(6).tolist(),
            "random_y": (grid * y[-1]).round(6).tolist(), "auuc": round(float(trapz(gy, grid)), 6)}


def validated_curve(model, k, points=101):
    """Exact curve on the held-out test set, using the synthetic true effect as the gain."""
    ite = pd.read_parquet(PROCESSED / "ite_test.parquet").drop(columns=["user_id"], errors="ignore")
    df = pd.read_parquet(PROCESSED / "test_metadata.parquet").merge(ite, on="row_id")
    gain = df[f"true_ite_{k}"].to_numpy()
    score = df[f"{MODEL_TAG[model]}_ite_{k}"].to_numpy()
    grid = np.linspace(0, 1, points)

    def curve(s):
        order = np.argsort(-s, kind="stable")
        n = len(order)
        y = np.r_[0.0, np.cumsum(gain[order]) / n]
        x = np.r_[0.0, np.arange(1, n + 1) / n]
        return np.interp(grid, x, y)

    my, py, ry = curve(score), curve(gain), grid * gain.mean()
    return {"x": grid.round(4).tolist(), "model_y": my.round(6).tolist(), "perfect_y": py.round(6).tolist(),
            "random_y": ry.round(6).tolist(), "auuc": float(trapz(my, grid)), "qini": float(trapz(my - ry, grid))}


def model_compare():
    out = {"accuracy": [], "qini": []}
    p, q = DOCS / "dml_results.csv", DOCS / "qini_summary.csv"
    if p.exists():
        out["accuracy"] = _clean(pd.read_csv(p))
    if q.exists():
        d = pd.read_csv(q)
        out["qini"] = _clean(d[d["view"] == "oracle"])
    return out


def propensity_info():
    p, m = PROCESSED / "propensity_train.parquet", PROCESSED / "train_metadata.parquet"
    if not (p.exists() and m.exists()):
        raise FileNotFoundError("Propensity files not found")
    df = pd.read_parquet(p).merge(pd.read_parquet(m)[["row_id", "discount_amount"]], on="row_id")
    edges = np.linspace(df["e_20"].min(), df["e_20"].max(), 31)
    groups = {}
    for g in (0, 10, 20):
        h, _ = np.histogram(df.loc[df["discount_amount"] == g, "e_20"], bins=edges, density=True)
        groups[str(g)] = h.round(4).tolist()
    out = {"edges": edges.round(4).tolist(), "groups": groups,
           "worst_before": None, "worst_after": None, "balance": []}
    f = DOCS / "propensity_balance.csv"
    if f.exists():
        b = pd.read_csv(f)
        b.columns = ["feature", "b10", "a10", "b20", "a20"]
        out["worst_before"] = float(b[["b10", "b20"]].abs().max().max())
        out["worst_after"] = float(b[["a10", "a20"]].abs().max().max())
        top = b.assign(w=b[["b10", "b20"]].abs().max(axis=1)).sort_values("w", ascending=False).head(12).drop(columns="w")
        out["balance"] = _clean(top.round(3))
    return out


def health_checks():
    items = []
    add = lambda n, s, d: items.append({"name": n, "status": s, "detail": d})
    ok = MODEL_FILE.exists()
    add("DML model bundle", "ok" if ok else "error", "models/dml_models.joblib" if ok else "Run src/causal/dml_model.py")
    p = DOCS / "dml_results.csv"
    if p.exists():
        d = pd.read_csv(p)
        r = d[(d["model"] == "CausalForestDML") & (d["discount"] == "$20")]
        if len(r):
            c = float(r["ITE correlation"].iloc[0])
            add("Causal validation", "ok" if c >= 0.7 else "warn", f"CausalForestDML ITE correlation {c:.2f} on held-out test data")
    else:
        add("Causal validation", "pending", "Run src/causal/dml_model.py")
    b = DOCS / "propensity_balance.csv"
    if b.exists():
        w = float(pd.read_csv(b).filter(like="after").abs().max().max())
        add("Covariate balance", "ok" if w < 0.1 else "warn", f"worst |SMD| after weighting {w:.3f} (target below 0.10)")
    else:
        add("Covariate balance", "pending", "Run src/data/propensity.py")
    add("Uplift evaluation", "ok" if (DOCS / "qini_summary.csv").exists() else "pending",
        "Qini and AUUC computed" if (DOCS / "qini_summary.csv").exists() else "Run src/causal/evaluate_uplift.py")
    r = DOCS / "refutation_results.csv"
    if r.exists():
        res = pd.read_csv(r)["result"].astype(str)
        add("Refutation audit", "warn" if (res == "FAIL").any() else "ok",
            f"{int((res == 'PASS').sum())} passed, {int((res == 'FAIL').sum())} failed")
    else:
        add("Refutation audit", "pending", "Not run yet (src/causal/refute.py)")
    return items


def drift_report(raw):
    global _reference
    if _reference is None:
        _reference = pd.read_csv(RAW)
    rows = []
    for col in NUMERIC_FEATURES:
        a = pd.to_numeric(_reference[col], errors="coerce").dropna()
        b = pd.to_numeric(raw[col], errors="coerce").dropna()
        if len(b) < 30:
            continue
        res = ks_2samp(a, b)
        rows.append({"feature": col, "test": "KS", "statistic": round(float(res.statistic), 4),
                     "p_value": float(res.pvalue),
                     "drifted": bool(res.pvalue < 0.01 and res.statistic > 0.1)})
    return rows


def customers(ds, model, category, age_group, search, sort, desc, page, page_size, profit_per_sale):
    sc = ds["scores"][model].copy()
    sc["segment"] = segments(sc).to_numpy()
    sc["recommended_discount"] = recommend(sc, (10, 20), profit_per_sale)
    raw = ds["raw"]
    has_age = "age_group" in raw.columns
    if has_age:
        sc["age_group"] = raw["age_group"].astype(str).to_numpy()
    mask = pd.Series(True, index=sc.index)
    if category and category != "All":
        mask &= sc["segment"] == category
    if has_age and age_group and age_group != "All":
        mask &= sc["age_group"] == age_group
    if search:
        mask &= sc["user_id"].astype(str).str.contains(search, regex=False)
    f = sc[mask]
    if sort not in ("ite_20", "ite_10", "p0", "user_id", "recommended_discount"):
        sort = "ite_20"
    f = f.sort_values(sort, ascending=not desc)
    page, page_size = max(1, page), min(max(1, page_size), 200)
    cols = [c for c in ("user_id", "age_group", "segment", "p0", "ite_10", "ite_20", "recommended_discount") if c in f.columns]
    total = int(len(f))
    return {"total": total, "page": page, "page_size": page_size,
            "rows": f.iloc[(page - 1) * page_size: page * page_size][cols].round(5).to_dict("records"),
            "kpis": {"filtered": total, "persuadables": int((f["segment"] == "Persuadable").sum()),
                     "avg_ite_20": float(f["ite_20"].mean()) if total else 0.0,
                     "recommended": int((f["recommended_discount"] > 0).sum())},
            "age_groups": sorted(raw["age_group"].astype(str).unique().tolist()) if has_age else []}