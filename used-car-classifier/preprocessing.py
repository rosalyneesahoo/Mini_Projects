from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "Age", "Kilometers_Driven", "Selling_Price", "Present_Price",
    "Fuel_Type", "Transmission", "Owner_Count", "Engine_CC", "Mileage",
    "Service_History", "Accident_History",
]
NUMERIC_FEATURES = [
    "Age", "Kilometers_Driven", "Selling_Price", "Present_Price",
    "Owner_Count", "Engine_CC", "Mileage",
]
CATEGORICAL_FEATURES = ["Fuel_Type", "Transmission", "Service_History", "Accident_History"]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
