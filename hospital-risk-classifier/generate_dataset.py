from pathlib import Path
import numpy as np
import pandas as pd


SEED = 42
N_ROWS = 650
OUTPUT = Path(__file__).parent / "data" / "patient_data.csv"

def generate_dataset(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 91, n_rows)
    gender = rng.choice(["Female", "Male", "Other"], n_rows, p=[0.48, 0.48, 0.04])
    heart_rate = np.clip(rng.normal(78, 15, n_rows), 42, 145).round(1)
    systolic = np.clip(rng.normal(125, 23, n_rows), 75, 210).round(1)
    diastolic = np.clip(rng.normal(79, 14, n_rows), 45, 135).round(1)
    temperature = np.clip(rng.normal(36.9, 0.55, n_rows), 35.2, 40.5).round(1)
    oxygen = np.clip(rng.normal(96.2, 2.3, n_rows), 84, 100).round(1)
    respiratory = np.clip(rng.normal(17.5, 4, n_rows), 8, 38).round(1)
    bmi = np.clip(rng.normal(26.5, 5.5, n_rows), 15, 48).round(1)
    previous = rng.choice(["No", "Yes"], n_rows, p=[0.68, 0.32])
    chronic = rng.choice(["No", "Yes"], n_rows, p=[0.62, 0.38])
    adherence = rng.choice(["Good", "Moderate", "Poor"], n_rows, p=[0.48, 0.37, 0.15])
    score = (
        0.035 * (age - 45)
        + 0.055 * np.abs(heart_rate - 75)
        + 0.035 * np.abs(systolic - 120)
        + 0.025 * np.abs(diastolic - 80)
        + 1.6 * np.maximum(0, 94 - oxygen)
        + 0.20 * np.abs(respiratory - 17)
        + 0.06 * np.abs(temperature - 36.8) * 10
        + 0.04 * np.abs(bmi - 24)
        + 1.8 * (previous == "Yes")
        + 1.7 * (chronic == "Yes")
        + 1.5 * (adherence == "Poor")
        + 0.7 * (adherence == "Moderate")
        + rng.normal(0, 2.7, n_rows)
    )
    risk_level = pd.qcut(score, q=[0, 0.48, 0.80, 1], labels=["LOW", "MEDIUM", "HIGH"]).astype(str)
    df = pd.DataFrame({
        "Patient_ID": [f"P{10001 + i}" for i in range(n_rows)], "Age": age,
        "Gender": gender, "Heart_Rate": heart_rate, "Systolic_BP": systolic,
        "Diastolic_BP": diastolic, "Temperature": temperature,
        "Oxygen_Saturation": oxygen, "Respiratory_Rate": respiratory, "BMI": bmi,
        "Previous_Hospitalization": previous, "Chronic_Condition": chronic,
        "Medication_Adherence": adherence, "Risk_Level": risk_level,
    })

    for column, fraction in {"Heart_Rate": 0.018, "Systolic_BP": 0.015, "Oxygen_Saturation": 0.018, "BMI": 0.015, "Medication_Adherence": 0.015}.items():
        indexes = rng.choice(n_rows, max(1, int(n_rows * fraction)), replace=False)
        df.loc[indexes, column] = np.nan
    return df


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
data = generate_dataset()
data.to_csv(OUTPUT, index=False)
print(f"Saved {len(data)} rows to {OUTPUT}")
print("Risk distribution:")
print(data["Risk_Level"].value_counts().sort_index())
print("Missing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])
