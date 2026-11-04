"""
propensity.py  -  Task 2.4 : propensity scores, overlap and balance check.

A propensity score answers:  "Given this customer's features, how likely was
it that marketing gave them $0, $10 or $20?"

We use it to check two things the causal model needs:
  1. OVERLAP  - every kind of customer had a real chance of each discount
  2. BALANCE  - after re-weighting, the discount groups look alike

Needs the files made by preprocess.py.
Run from anywhere:   python src/data/propensity.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw charts without a window
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS = PROJECT_ROOT / "docs"

TREATMENT = "discount_amount"
LEVELS = [0, 10, 20]
LOW_PROPENSITY = 0.05        # below this, overlap is doubtful
GOOD_SMD = 0.10              # |SMD| below this = groups are balanced


def load_data():
    """Read the files created by preprocess.py and join features with treatment."""
    X_train = pd.read_parquet(PROCESSED / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED / "X_test.parquet")
    meta_train = pd.read_parquet(PROCESSED / "train_metadata.parquet")
    meta_test = pd.read_parquet(PROCESSED / "test_metadata.parquet")

    # both files are sorted by row_id, so join on row_id to be safe
    train = X_train.merge(meta_train[["row_id", TREATMENT]], on="row_id", validate="one_to_one")
    test = X_test.merge(meta_test[["row_id", TREATMENT]], on="row_id", validate="one_to_one")

    # features = every column except the id and the treatment
    features = [c for c in X_train.columns if c != "row_id"]
    return train, test, features


def smd(a, b, weights_a=None, weights_b=None):
    """Standardized mean difference between two groups (optionally weighted)."""
    wa = np.ones(len(a)) if weights_a is None else weights_a
    wb = np.ones(len(b)) if weights_b is None else weights_b
    mean_a, mean_b = np.average(a, weights=wa), np.average(b, weights=wb)
    var_a = np.average((a - mean_a) ** 2, weights=wa)
    var_b = np.average((b - mean_b) ** 2, weights=wb)
    if var_a + var_b == 0:
        return 0.0
    return (mean_a - mean_b) / np.sqrt((var_a + var_b) / 2)


def main():
    train, test, features = load_data()
    print("Step 1  Loaded: train", train.shape, "| test", test.shape,
          "| features:", len(features))

    # ---------------- Step 2: fit the propensity model on TRAIN only ----------
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[features], train[TREATMENT])
    classes = list(model.classes_)                 # [0, 10, 20]

    def scores(frame):
        p = pd.DataFrame(model.predict_proba(frame[features]),
                         columns=[f"e_{c}" for c in classes])
        p.insert(0, "row_id", frame["row_id"].to_numpy())
        return p

    p_train, p_test = scores(train), scores(test)
    print("Step 2  Propensity model fitted")

    # ---------------- Step 3: OVERLAP check ----------------
    print("\nStep 3  OVERLAP  (smallest / median / largest chance of each discount)")
    for c in classes:
        col = p_train[f"e_{c}"]
        print(f"   P(discount = ${c:<2}) : {col.min():.3f} / {col.median():.3f} / {col.max():.3f}")
    n_low = int((p_train[[f"e_{c}" for c in classes]].min(axis=1) < LOW_PROPENSITY).sum())
    print(f"   Customers with a chance below {LOW_PROPENSITY}: {n_low} of {len(p_train):,}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, c in zip(axes, classes):
        for got in classes:
            mask = (train[TREATMENT] == got).to_numpy()
            ax.hist(p_train.loc[mask, f"e_{c}"], bins=40, alpha=0.5, density=True,
                    label=f"customers who got ${got}")
        ax.set_xlabel(f"Estimated chance of getting ${c}")
        ax.set_ylabel("Density of customers")
        ax.set_title(f"Overlap for ${c} discount")
        ax.legend(fontsize=8)
    plt.tight_layout()
    DOCS.mkdir(exist_ok=True)
    plt.savefig(DOCS / "propensity_overlap.png", dpi=120)
    plt.close()

    # ---------------- Step 4: BALANCE before / after weighting ----------------
    t = train[TREATMENT].to_numpy()
    share = pd.Series(t).value_counts(normalize=True)          # overall share of each discount
    own_chance = np.array([p_train.loc[i, f"e_{t[i]}"] for i in range(len(t))])
    weights = np.array([share[v] for v in t]) / own_chance      # inverse-propensity weights
    weights = np.minimum(weights, np.quantile(weights, 0.99))   # cap extreme weights
    print(f"\nStep 4  Weights: average {weights.mean():.2f}, largest {weights.max():.2f}")

    rows = []
    for f in features:
        values = train[f].to_numpy(dtype=float)
        row = {"feature": f}
        for g in (10, 20):
            a, b = (t == g), (t == 0)
            row[f"${g} vs $0 before"] = smd(values[a], values[b])
            row[f"${g} vs $0 after"] = smd(values[a], values[b], weights[a], weights[b])
        rows.append(row)

    balance = pd.DataFrame(rows).set_index("feature").round(3)
    DOCS.mkdir(exist_ok=True)
    balance.to_csv(DOCS / "propensity_balance.csv")

    before = balance[[c for c in balance if "before" in c]].abs().max().max()
    after = balance[[c for c in balance if "after" in c]].abs().max().max()
    print(f"        Worst |SMD| before weighting: {before:.3f}")
    print(f"        Worst |SMD| after  weighting: {after:.3f}   (below {GOOD_SMD} is good)")

    top = balance.abs().max(axis=1).sort_values(ascending=False).head(5).index
    print("\n        Features with the biggest imbalance before weighting:")
    print(balance.loc[top].to_string())

    # ---------------- Step 5: save the scores ----------------
    p_train.to_parquet(PROCESSED / "propensity_train.parquet", index=False)
    p_test.to_parquet(PROCESSED / "propensity_test.parquet", index=False)
    print("\nStep 5  Saved: data/processed/propensity_train.parquet, propensity_test.parquet,")
    print("        docs/propensity_overlap.png, docs/propensity_balance.csv")


if __name__ == "__main__":
    main()