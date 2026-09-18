from pathlib import Path
import argparse
import joblib
import pandas as pd


FEATURES = ["Age", "Annual_Income", "Total_Purchases", "Average_Order_Value",
            "Purchase_Frequency", "Total_Spending", "Discount_Usage",
            "Online_Purchases", "Store_Purchases"]
MODEL_DIR = Path("models")

def predict_customer(values):
    if len(values) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} values in this order: {', '.join(FEATURES)}")
    row = pd.DataFrame([values], columns=FEATURES)
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model = joblib.load(MODEL_DIR / "knn_model.pkl")
    return model.predict(scaler.transform(imputer.transform(row)))[0]


parser = argparse.ArgumentParser(description="Classify one retail customer with KNN.")
parser.add_argument("values", nargs="*", type=float, help="9 feature values in the order shown below")
args = parser.parse_args()
values = args.values
if not values:
    print("Enter values in this order:")
    for feature in FEATURES:
        values.append(float(input(f"{feature}: ")))
print("\n--------------------------------")
print("CUSTOMER CLASSIFICATION")
print("--------------------------------")
print("Predicted Customer Class:")
print(predict_customer(values))
