"""
evaluate_uplift.py  -  Task 2.7 : Qini / uplift curves and AUUC.

Question answered:
  "If we give the discount only to the customers a method ranks highest, how many
   EXTRA purchases do we win compared with giving it to random customers?"

Methods compared
  CausalForestDML, LinearDML, T-Learner   rank by predicted discount effect (ITE)
  Plain P(buy) model                       ordinary ML: rank by chance of buying, ignores the discount
  Random                                   the straight diagonal line
  Perfect (true ITE)                       best possible ranking (synthetic data only)

Two views of the same curves
  oracle        gain = true ITE from the answer key              (synthetic data only)
  observed_ipw  gain = purchases re-weighted by propensity scores (works on REAL data too)

How to read a curve: x = share of customers given the discount (best-ranked first),
y = extra purchases per customer so far. Above the diagonal = the method helps.

Needs the files from preprocess.py, propensity.py and dml_model.py.
Run from anywhere:   python src/causal/evaluate_uplift.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
PROCESSED, DOCS = ROOT / "data" / "processed", ROOT / "docs"
trapz = getattr(np, "trapezoid", None) or np.trapz

LEVELS = [10, 20]
MODELS = {"CausalForestDML": "causalforestdml", "LinearDML": "lineardml", "T-Learner": "tlearner"}
PLAIN, PERFECT = "Plain P(buy) model", "Perfect (true ITE)"
MIN_PROPENSITY = 0.05          # stops a few customers from getting huge weights
COLORS = {"CausalForestDML": "tab:green", "LinearDML": "tab:orange", "T-Learner": "tab:blue",
          PLAIN: "tab:purple", PERFECT: "black"}


# ------------------------------------------------------------------
# CURVE HELPERS
# ------------------------------------------------------------------
def gain_curve(score, gain):
    """Sort customers by score (best first) and add up their gains. Returns x, y."""
    order = np.argsort(-np.asarray(score), kind="stable")
    n = len(order)
    y = np.r_[0.0, np.cumsum(np.asarray(gain)[order]) / n]
    x = np.r_[0.0, np.arange(1, n + 1) / n]
    return x, y


def summarize(x, y):
    """AUUC = area under the curve. Qini coefficient = area ABOVE the random diagonal."""
    return trapz(y, x), trapz(y - x * y[-1], x)


def on_grid(x, y, points=101):
    grid = np.linspace(0, 1, points)
    return grid, np.interp(grid, x, y)


def ipw_gain(df, k):
    """
    One number per customer whose AVERAGE is the customer's effect of discount k.
      got $k and bought :  +1 / chance of getting $k
      got $0 and bought :  -1 / chance of getting $0
      anyone else       :   0
    Dividing by the chance removes the bias from marketing choosing who got what.
    """
    y, t = df["purchased"].to_numpy(float), df["discount_amount"].to_numpy()
    e_k = df[f"e_{k}"].clip(lower=MIN_PROPENSITY).to_numpy()
    e_0 = df["e_0"].clip(lower=MIN_PROPENSITY).to_numpy()
    return y * ((t == k) / e_k - (t == 0) / e_0)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    DOCS.mkdir(exist_ok=True)
    X_tr = pd.read_parquet(PROCESSED / "X_train.parquet")
    m_tr = pd.read_parquet(PROCESSED / "train_metadata.parquet")
    X_te = pd.read_parquet(PROCESSED / "X_test.parquet")
    m_te = pd.read_parquet(PROCESSED / "test_metadata.parquet")
    ite = pd.read_parquet(PROCESSED / "ite_test.parquet").drop(columns=["user_id"])
    prop = pd.read_parquet(PROCESSED / "propensity_test.parquet")

    df = (m_te.merge(ite, on="row_id", validate="one_to_one")
              .merge(prop, on="row_id", validate="one_to_one"))
    X_te = X_te.set_index("row_id").loc[df["row_id"]]          # same customer order as df
    feats = [c for c in X_tr.columns if c != "row_id"]
    print(f"Step 1  Loaded {len(df):,} test customers")

    # ordinary ML baseline: learns who buys, never sees the discount
    plain = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                           min_child_samples=50, random_state=42, verbose=-1)
    plain.fit(X_tr[feats], m_tr["purchased"])
    df[PLAIN] = plain.predict_proba(X_te[feats])[:, 1]
    print("Step 2  Plain P(buy) model trained")

    rows, true_curves, dashboard, dashboard_true = [], {}, {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    for r, view in enumerate(["oracle", "observed_ipw"]):
        for c, k in enumerate(LEVELS):
            gain = df[f"true_ite_{k}"].to_numpy() if view == "oracle" else ipw_gain(df, k)

            scores = {name: df[f"{tag}_ite_{k}"].to_numpy() for name, tag in MODELS.items()}
            scores[PLAIN] = df[PLAIN].to_numpy()
            if view == "oracle":
                scores[PERFECT] = df[f"true_ite_{k}"].to_numpy()

            ax = axes[r, c]
            for name, s in scores.items():
                x, y = gain_curve(s, gain)
                auuc, qini = summarize(x, y)
                rows.append({"view": view, "discount": f"${k}", "method": name,
                             "AUUC": auuc, "Qini coefficient": qini})
                ax.plot(x, y * 100, "--" if name == PERFECT else "-", color=COLORS[name], label=name)
                if name == "CausalForestDML":
                    gx, gy = on_grid(x, y)
                    payload = {"x": gx.round(4).tolist(), "model_y": gy.round(6).tolist(),
                               "random_y": (gx * y[-1]).round(6).tolist(), "auuc": round(float(auuc), 6)}
                    (dashboard if view == "observed_ipw" else dashboard_true)[f"discount_{k}"] = payload
                    if view == "oracle":
                        true_curves[k] = (x, y, gain)

            ax.plot([0, 1], [0, gain.mean() * 100], ":", color="grey", label="Random")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"\\${k} vs \\$0  ({view})")   # \$ stops matplotlib reading $...$ as math
            ax.set_xlabel("Share of customers given the discount (best-ranked first)")
            ax.set_ylabel("Extra purchases per 100 customers")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(DOCS / "qini_curves.png", dpi=130)
    plt.close()
    print("Step 3  Curves drawn")

    table = pd.DataFrame(rows).round(5)
    table.to_csv(DOCS / "qini_summary.csv", index=False)
    for view in ["oracle", "observed_ipw"]:
        pivot = (table[table["view"] == view]
                 .pivot(index="method", columns="discount", values="Qini coefficient") * 100)
        print(f"\nQINI COEFFICIENT, view = {view}  (extra purchases per 100 customers above random)")
        print(pivot.sort_values("$20", ascending=False).round(3).to_string())

    # ---- business view (exact, uses the true gain) ----
    view_rows = []
    for k in LEVELS:
        x, y, gain = true_curves[k]
        n = len(gain)
        peak = int(np.argmax(y))
        view_rows.append({"discount": f"${k}",
                          "give to everyone": gain.mean() * 1000,
                          "best 50% (CausalForestDML)": y[int(0.5 * n)] * 1000,
                          "best 30% (CausalForestDML)": y[int(0.3 * n)] * 1000,
                          "best share to target (%)": x[peak] * 100,
                          "extra purchases at that share": y[peak] * 1000})
    business = pd.DataFrame(view_rows).round(1)
    business.to_csv(DOCS / "qini_business_view.csv", index=False)
    print("\nBUSINESS VIEW  (extra purchases per 1,000 customers)")
    print(business.to_string(index=False))

    # ---- dashboard files (shape from the API contract: x, model_y, random_y, auuc) ----
    (DOCS / "qini_curves.json").write_text(json.dumps(dashboard))              # works on real data
    (DOCS / "qini_curves_true.json").write_text(json.dumps(dashboard_true))    # synthetic only
    print("\nSaved: docs/qini_curves.png, qini_summary.csv, qini_business_view.csv,")
    print("       qini_curves.json (real-data safe), qini_curves_true.json (synthetic only)")


if __name__ == "__main__":
    main()