from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_RECORDS = 400
OUTPUT = Path(__file__).parent / "data" / "used_cars.csv"

def generate_dataset(n_records=N_RECORDS, seed=SEED):
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 16, n_records)
    km = np.clip(rng.normal(45000 + age * 6500, 22000, n_records), 5000, 180000).round().astype(int)
    present_price = np.clip(rng.normal(850000, 350000, n_records), 250000, 2200000).round(-3)
    depreciation = np.clip(0.12 * age + km / 1500000 + rng.normal(0, 0.04, n_records), 0.08, 0.82)
    selling_price = np.clip(present_price * (1 - depreciation) + rng.normal(0, 45000, n_records), 80000, 2000000).round(-3)
    fuel = rng.choice(["Petrol", "Diesel", "CNG"], n_records, p=[0.55, 0.35, 0.10])
    transmission = rng.choice(["Manual", "Automatic"], n_records, p=[0.70, 0.30])
    owners = np.clip(rng.poisson(0.8, n_records) + 1, 1, 4)
    engine = np.clip(rng.normal(1350, 400, n_records), 800, 3000).round().astype(int)
    mileage = np.clip(24 - age * 0.35 - engine / 1800 + rng.normal(0, 2.2, n_records), 8, 30).round(1)
    service = rng.choice(["Good", "Average", "Poor"], n_records, p=[0.50, 0.35, 0.15])
    accident = rng.choice(["No", "Yes"], n_records, p=[0.78, 0.22])

    score = (
        2.0 - age * 0.15 - km / 120000
        + (selling_price / np.maximum(present_price, 1)) * 1.8
        + np.where(service == "Good", 1.3, np.where(service == "Average", 0.3, -1.2))
        + np.where(accident == "No", 0.9, -1.1)
        - (owners - 1) * 0.55 + mileage * 0.06
        + rng.normal(0, 1.15, n_records)
    )
    suitability = np.where(score >= np.median(score), "SUITABLE", "NOT_SUITABLE")

    df = pd.DataFrame({
        "Car_ID": np.arange(1001, 1001 + n_records), "Age": age,
        "Kilometers_Driven": km, "Selling_Price": selling_price,
        "Present_Price": present_price, "Fuel_Type": fuel,
        "Transmission": transmission, "Owner_Count": owners,
        "Engine_CC": engine, "Mileage": mileage,
        "Service_History": service, "Accident_History": accident,
        "Purchase_Suitability": suitability,
    })
    for column, fraction in {"Mileage": 0.025, "Engine_CC": 0.02, "Service_History": 0.02, "Selling_Price": 0.02}.items():
        indexes = rng.choice(df.index, size=max(1, int(n_records * fraction)), replace=False)
        df.loc[indexes, column] = np.nan

    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
data = generate_dataset()
data.to_csv(OUTPUT, index=False)
print(f"Saved {len(data)} rows to {OUTPUT}")
print(data["Purchase_Suitability"].value_counts().to_string())