from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "loan_approved"
NUMERICAL_FEATURES = [
    "age", "annual_income", "credit_score", "loan_amount", "loan_term_months",
    "employment_years", "debt_to_income", "existing_loans", "savings_balance", "dependents",
]
CATEGORICAL_FEATURES = ["home_ownership", "education", "marital_status", "purpose"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def make_preprocessor() -> ColumnTransformer:
    """Return the one preprocessor used both during training and prediction."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)

def load_data(data_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(data_path)
    missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].astype(int).copy()

def fit_transform_data(X: pd.DataFrame):
    preprocessor = make_preprocessor()
    transformed = preprocessor.fit_transform(X)
    return transformed, preprocessor

def feature_names(preprocessor: ColumnTransformer):
    return list(preprocessor.get_feature_names_out())
