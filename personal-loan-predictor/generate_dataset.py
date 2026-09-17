from pathlib import Path
import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_ROWS = 700

def generate_dataset(n_rows=N_ROWS, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 61, n_rows)
    gender = rng.choice(["Male", "Female"], n_rows, p=[0.68, 0.32])
    married = rng.choice(["Yes", "No"], n_rows, p=[0.62, 0.38])
    education = rng.choice(["Graduate", "Not Graduate"], n_rows, p=[0.72, 0.28])
    self_employed = rng.choice(["Yes", "No"], n_rows, p=[0.14, 0.86])
    applicant_income = np.clip(rng.lognormal(np.log(55_000), 0.48, n_rows), 12_000, 250_000).round(0)
    coapplicant_income = np.where(
        married == "Yes",
        np.clip(rng.lognormal(np.log(22_000), 0.65, n_rows), 0, 130_000),
        rng.choice([0, 0, 0, 8_000, 15_000], n_rows),
    ).round(0)
    loan_amount = np.clip(rng.lognormal(np.log(280_000), 0.48, n_rows), 50_000, 1_000_000).round(0)
    loan_term = rng.choice([120, 180, 240, 300, 360, 480], n_rows, p=[0.04, 0.08, 0.12, 0.16, 0.50, 0.10])
    credit_history = rng.choice(["Good", "Poor"], n_rows, p=[0.78, 0.22])
    existing_loans = np.clip(rng.poisson(1.1, n_rows), 0, 5)
    employment_status = rng.choice(["Salaried", "Self-Employed", "Contract", "Unemployed"], n_rows, p=[0.52, 0.18, 0.22, 0.08])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n_rows, p=[0.38, 0.37, 0.25])
    total_income = applicant_income + coapplicant_income
    debt_to_income_ratio = np.clip(
        (loan_amount / np.maximum(total_income, 1)) * (360 / loan_term) * 0.24
        + rng.normal(0, 0.055, n_rows),
        0.05,
        0.95,
    ).round(3)

    score = (
        1.15 * (credit_history == "Good")
        + 0.55 * (education == "Graduate")
        + 0.35 * (employment_status == "Salaried")
        + 0.22 * (employment_status == "Self-Employed")
        + 0.18 * (employment_status == "Contract")
        + 0.22 * (property_area == "Urban")
        + 0.10 * (property_area == "Semiurban")
        + 0.30 * np.clip(np.log1p(total_income) - 10.3, -1, 2)
        - 1.75 * debt_to_income_ratio
        - 0.23 * existing_loans
        - 0.18 * (loan_amount / 300_000)
        + rng.normal(0, 0.65, n_rows)
    )
    threshold = np.quantile(score, 0.43)
    eligibility = np.where(score >= threshold, "ELIGIBLE", "NOT_ELIGIBLE")

    df = pd.DataFrame({
        "Applicant_ID": [f"APP{idx:04d}" for idx in range(1, n_rows + 1)],
        "Age": age,
        "Gender": gender,
        "Married": married,
        "Education": education,
        "Self_Employed": self_employed,
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Credit_History": credit_history,
        "Existing_Loans": existing_loans,
        "Debt_to_Income_Ratio": debt_to_income_ratio,
        "Employment_Status": employment_status,
        "Property_Area": property_area,
        "Loan_Eligibility": eligibility,
    })

    missing_rates = {
        "Gender": 0.015, "Education": 0.012, "Self_Employed": 0.015,
        "Applicant_Income": 0.012, "Loan_Amount": 0.012,
        "Credit_History": 0.015, "Existing_Loans": 0.01,
        "Debt_to_Income_Ratio": 0.012, "Employment_Status": 0.012,
    }
    for column, rate in missing_rates.items():
        mask = rng.random(n_rows) < rate
        df.loc[mask, column] = np.nan
    return df

if __name__ == "__main__":
    output_path = Path(__file__).parent / "data" / "loan_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(output_path, index=False)
    print(f"Generated {len(dataset)} rows at {output_path}")
    print(dataset["Loan_Eligibility"].value_counts())
    print("Missing values:")
    print(dataset.isna().sum()[dataset.isna().sum() > 0])
