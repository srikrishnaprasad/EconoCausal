"""
EconoCausal — Data Preprocessing
File: src/data/preprocess.py

Responsibilities:
- Load the raw retail dataset.
- Define treatment (discount_amount), outcome (purchased), and pre-treatment features.
- Split records into training and test sets.
- Fit imputers/encoders on training data only.
- Preserve row_id, user_id, treatment, outcome, and available synthetic ground truth.
- Save processed feature matrices and metadata as Parquet files.
- Save the fitted preprocessor for reuse on new data.

Important:
- Ground-truth columns are for validation only and are never model features.
- Verify that income_bracket is an ordered numeric level before treating it as numeric.
- Confirm that every feature is measured before discount assignment.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PATHS
# ============================================================

# This file is expected at <project_root>/src/data/preprocess.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Retail_synthetic_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# 2. CAUSAL ROLES AND FEATURE DEFINITIONS
# ============================================================

TREATMENT_COL = "discount_amount"
OUTCOME_COL = "purchased"
ID_COL = "user_id"
ROW_ID_COL = "row_id"

# Numeric features.
# income_bracket is treated as numeric ONLY under the assumption that
# 1..12 encode ordered income levels. Move it to CATEGORICAL_FEATURES
# if these are arbitrary category codes instead.
NUMERIC_FEATURES = [
    "income_bracket",
    "is_new_customer",
    "buys_mens",
    "buys_womens",
    "months_since_last_purchase",
    "past_spend",
    "site_visits_90d",
    "loyalty_score",
    "coupon_usage_rate",
]

# Nominal categories are one-hot encoded.
CATEGORICAL_FEATURES = [
    "age_group",
    "region_type",
    "purchase_channel",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# These columns are preserved for validation/reporting only.
# They MUST NOT be added to FEATURE_COLS.
GROUND_TRUTH_COLS = [
    "true_p_buy_0",
    "true_p_buy_10",
    "true_p_buy_20",
    "true_ite_10",
    "true_ite_20",
    "true_customer_type",
]


# ============================================================
# 3. PREPROCESSOR FACTORY
# ============================================================

def make_preprocessor():
    """Create a new preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ============================================================
# 4. VALIDATE AND SELECT FEATURES
# ============================================================

def prepare_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features in a fixed order and validate their presence."""

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            "Required feature columns are missing from the dataset: "
            f"{missing}"
        )

    X = df[FEATURE_COLS].copy()

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="raise")

    return X


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate required treatment/outcome/ID columns and create a stable row_id.
    Rows missing treatment or outcome are excluded; they cannot safely be
    imputed.
    """

    required = [ID_COL, TREATMENT_COL, OUTCOME_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    data = df.copy().reset_index(drop=True)

    # Avoid overwriting a source column called row_id.
    if ROW_ID_COL in data.columns:
        data = data.rename(columns={ROW_ID_COL: "source_row_id"})

    data.insert(0, ROW_ID_COL, np.arange(len(data), dtype=np.int64))

    valid_rows = data[TREATMENT_COL].notna() & data[OUTCOME_COL].notna()
    data = data.loc[valid_rows].copy().reset_index(drop=True)

    if not data[TREATMENT_COL].isin([0, 10, 20]).all():
        unexpected = sorted(
            data.loc[
                ~data[TREATMENT_COL].isin([0, 10, 20]),
                TREATMENT_COL,
            ].unique().tolist()
        )
        raise ValueError(
            "discount_amount must contain only 0, 10, 20. "
            f"Unexpected values: {unexpected}"
        )

    if not data[OUTCOME_COL].isin([0, 1]).all():
        unexpected = sorted(
            data.loc[
                ~data[OUTCOME_COL].isin([0, 1]),
                OUTCOME_COL,
            ].unique().tolist()
        )
        raise ValueError(
            "purchased must contain only 0 and 1. "
            f"Unexpected values: {unexpected}"
        )

    counts = data[TREATMENT_COL].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            "Each treatment level needs at least two valid rows "
            "for a stratified train/test split."
        )

    return data


# ============================================================
# 5. MAIN PREPROCESSING FUNCTION
# ============================================================

def build_processed_dataset(df: pd.DataFrame,test_size: float = 0.20,random_state: int = 42,save: bool = True,) -> Dict[str, Any]:
    """
    Split and preprocess the dataset.

    Returns:
        X_train, X_test: named encoded feature DataFrames with row_id.
        train_metadata, test_metadata: identity, treatment, outcome,
            and available ground-truth columns.
        preprocessor: fitted sklearn preprocessing pipeline.

    If save=True, outputs are written to data/processed/.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = validate_dataset(df)
    X_raw = prepare_raw_features(data)
    T = data[TREATMENT_COL].astype(int)

    metadata_cols = [ROW_ID_COL,ID_COL,TREATMENT_COL,OUTCOME_COL]

    if "source_row_id" in data.columns:
        metadata_cols.append("source_row_id")

    metadata_cols.extend(
        col for col in GROUND_TRUTH_COLS if col in data.columns
    )
    metadata = data[metadata_cols].copy()

    train_positions, test_positions = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=random_state,
        stratify=T,
    )

    X_train_raw = X_raw.iloc[train_positions].copy()
    X_test_raw = X_raw.iloc[test_positions].copy()

    # Fit only on training features to prevent leakage.
    preprocessor = make_preprocessor()
    X_train_values = preprocessor.fit_transform(X_train_raw)
    X_test_values = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(X_train_values, columns=feature_names)
    X_train.insert(
        0, ROW_ID_COL, data.iloc[train_positions][ROW_ID_COL].to_numpy()
    )

    X_test = pd.DataFrame(X_test_values, columns=feature_names)
    X_test.insert(
        0, ROW_ID_COL, data.iloc[test_positions][ROW_ID_COL].to_numpy()
    )

    train_metadata = metadata.iloc[train_positions].reset_index(drop=True)
    test_metadata = metadata.iloc[test_positions].reset_index(drop=True)

    X_train = X_train.sort_values(ROW_ID_COL).reset_index(drop=True)
    X_test = X_test.sort_values(ROW_ID_COL).reset_index(drop=True)
    train_metadata = train_metadata.sort_values(ROW_ID_COL).reset_index(drop=True)
    test_metadata = test_metadata.sort_values(ROW_ID_COL).reset_index(drop=True)

    # Confirm that feature rows and metadata rows map correctly.
    assert X_train[ROW_ID_COL].is_unique
    assert X_test[ROW_ID_COL].is_unique
    assert train_metadata[ROW_ID_COL].is_unique
    assert test_metadata[ROW_ID_COL].is_unique

    assert set(X_train[ROW_ID_COL]) == set(train_metadata[ROW_ID_COL])
    assert set(X_test[ROW_ID_COL]) == set(test_metadata[ROW_ID_COL])
    assert set(X_train[ROW_ID_COL]).isdisjoint(set(X_test[ROW_ID_COL]))

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
        X_test.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
        train_metadata.to_parquet(
            PROCESSED_DIR / "train_metadata.parquet", index=False
        )
        test_metadata.to_parquet(
            PROCESSED_DIR / "test_metadata.parquet", index=False
        )

        joblib.dump(
            preprocessor,
            PROCESSED_DIR / "preprocessor.joblib",
        )

        print(f"Processed files saved to: {PROCESSED_DIR}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
        "preprocessor": preprocessor,
    }


# ============================================================
# 6. REUSE THE FITTED PREPROCESSOR ON NEW CUSTOMER DATA
# ============================================================

def transform_new_data(
    new_df: pd.DataFrame,
    preprocessor=None,
    preprocessor_path: Path = None,
) -> pd.DataFrame:
    """
    Transform new customer rows using an already-fitted pipeline.

    Pass either a fitted preprocessor or a path to preprocessor.joblib.
    The pipeline is not refitted on new data.
    """

    if preprocessor is None:
        if preprocessor_path is None:
            preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"

        preprocessor_path = Path(preprocessor_path)
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Saved preprocessor not found: {preprocessor_path}"
            )

        preprocessor = joblib.load(preprocessor_path)

    X_new_raw = prepare_raw_features(new_df)
    values = preprocessor.transform(X_new_raw)
    feature_names = preprocessor.get_feature_names_out()

    X_new = pd.DataFrame(
        values,
        columns=feature_names,
        index=new_df.index,
    )
    X_new.index.name = "original_index"

    return X_new.reset_index()


# ============================================================
# 7. RUN THIS MODULE DIRECTLY
# ============================================================

if __name__ == "__main__":
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}\n"
            "Expected: data/raw/Retail_synthetic_dataset.csv"
        )

    raw_df = pd.read_csv(RAW_PATH)

    print("DATASET LOADED")
    print("Original rows:", len(raw_df))
    print("Original columns:", len(raw_df.columns))

    result = build_processed_dataset(raw_df, save=True)

    print("\n--- PREPROCESSING SUMMARY ---")
    print("Training feature shape:", result["X_train"].shape)
    print("Testing feature shape:", result["X_test"].shape)
    print("Training metadata shape:", result["train_metadata"].shape)
    print("Testing metadata shape:", result["test_metadata"].shape)

    print("\nTest metadata columns:")
    print(result["test_metadata"].columns.tolist())

    for col in ["true_ite_10", "true_ite_20", "true_customer_type"]:
        print(
            f"{col} preserved:",
            col in result["test_metadata"].columns,
        )

    print("\nPreprocessing completed successfully.")
