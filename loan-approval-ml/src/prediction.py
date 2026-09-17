from pathlib import Path
import json
import joblib
import pandas as pd


def sample_applicant() -> pd.DataFrame:
    return pd.DataFrame([{
        "age": 34, "annual_income": 78000, "credit_score": 735, "loan_amount": 24000,
        "loan_term_months": 36, "employment_years": 8, "debt_to_income": .22,
        "existing_loans": 1, "savings_balance": 28000, "home_ownership": "Mortgage",
        "education": "Bachelor", "marital_status": "Married", "dependents": 1,
        "purpose": "Home Improvement",
    }])

def predict_sample(artifact_dir: str | Path):
    artifact_dir = Path(artifact_dir)
    model = joblib.load(artifact_dir / "selected_model.joblib")
    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    applicant = sample_applicant()
    transformed = preprocessor.transform(applicant)
    prediction = int(model.predict(transformed)[0])
    probability = float(model.predict_proba(transformed)[0, 1]) if hasattr(model, "predict_proba") else None
    result = {"applicant": applicant.iloc[0].to_dict(), "prediction": prediction,
              "label": "Approved" if prediction else "Not approved", "approval_probability": probability}
    (artifact_dir / "sample_prediction.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Sample applicant: {result['label']}" + (f" (probability={probability:.1%})" if probability is not None else ""))
    return result

def main():
    predict_sample(Path(__file__).resolve().parents[1] / "artifacts")


main()
