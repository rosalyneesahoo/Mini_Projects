from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "loan_applications.csv"

def generate_dataset(n_rows: int = 800, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 66, n_rows)
    annual_income = np.clip(rng.lognormal(np.log(62000), 0.48, n_rows), 18000, 240000).round(0)
    credit_score = np.clip(rng.normal(680, 72, n_rows), 420, 830).round(0)
    employment_years = np.minimum(np.clip((age - 21) * rng.uniform(.35, .8, n_rows) + rng.normal(0, 2, n_rows), 0, 40), age - 18).round(1)
    loan_amount = np.clip(annual_income * rng.uniform(.12, .72, n_rows) + rng.normal(0, 9000, n_rows), 3000, 130000).round(0)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 72], n_rows, p=[.08, .14, .28, .18, .24, .08])
    debt_to_income = np.clip(rng.beta(2.5, 6, n_rows) * .72 + loan_amount / annual_income * .10 + rng.normal(0, .025, n_rows), .03, .82).round(3)
    existing_loans = np.clip(rng.poisson(1.4, n_rows), 0, 6)
    savings_balance = np.clip(annual_income * rng.beta(1.8, 7, n_rows) + rng.normal(0, 2500, n_rows), 0, 140000).round(0)
    home_ownership = rng.choice(["Rent", "Mortgage", "Own", "Other"], n_rows, p=[.38, .38, .18, .06])
    education = rng.choice(["High School", "Bachelor", "Master", "Doctorate"], n_rows, p=[.28, .42, .24, .06])
    marital_status = rng.choice(["Single", "Married", "Divorced"], n_rows, p=[.40, .47, .13])
    dependents = np.clip(rng.poisson(1.1, n_rows), 0, 5)
    purpose = rng.choice(["Car", "Home Improvement", "Education", "Medical", "Debt Consolidation", "Business"], n_rows, p=[.22, .18, .13, .10, .25, .12])

    logit = (
        -2.0 + 0.010 * (credit_score - 650) + 0.000008 * (annual_income - 60000)
        - 0.000014 * loan_amount - 2.2 * debt_to_income + 0.045 * employment_years
        + 0.000006 * savings_balance - 0.16 * existing_loans - 0.012 * dependents
        + np.where(home_ownership == "Own", .42, np.where(home_ownership == "Mortgage", .16, 0))
        + np.where(education == "Master", .18, np.where(education == "Doctorate", .30, 0))
        + np.where(purpose == "Debt Consolidation", -.18, 0)
        + np.where(purpose == "Business", -.08, 0)
        + rng.normal(0, .75, n_rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    loan_approved = rng.binomial(1, probability)

    frame = pd.DataFrame({
        "age": age, "annual_income": annual_income, "credit_score": credit_score,
        "loan_amount": loan_amount, "loan_term_months": loan_term_months,
        "employment_years": employment_years, "debt_to_income": debt_to_income,
        "existing_loans": existing_loans, "savings_balance": savings_balance,
        "home_ownership": home_ownership, "education": education,
        "marital_status": marital_status, "dependents": dependents, "purpose": purpose,
        "loan_approved": loan_approved,
    })
    missing_rates = {"annual_income": .035, "credit_score": .04, "loan_amount": .03,
                     "employment_years": .04, "debt_to_income": .035, "savings_balance": .04,
                     "home_ownership": .025, "education": .02, "purpose": .025}
    for column, rate in missing_rates.items():
        mask = rng.random(n_rows) < rate
        frame.loc[mask, column] = np.nan
    return frame

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset()
    frame.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(frame):,} rows at {DATA_PATH}")
    print("Approval rate: {:.1%}; missing cells: {}".format(frame.loan_approved.mean(), int(frame.isna().sum().sum())))


main()
