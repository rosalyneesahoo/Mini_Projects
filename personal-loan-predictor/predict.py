from pathlib import Path
import argparse
import joblib
import pandas as pd


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "final_model.pkl"

def predict_applicant(applicant):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    row = pd.DataFrame([applicant])
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    probability_map = dict(zip(model.classes_, probabilities))
    return bundle["model_name"], prediction, probability_map

def observations(applicant):
    notes = []
    notes.append("Good credit history is one characteristic considered by the model." if applicant["Credit_History"] == "Good" else "Poor credit history is one characteristic considered by the model.")
    notes.append("The employment category is treated as a stable employment indicator." if applicant["Employment_Status"] in ["Salaried", "Self-Employed"] else "The employment category may represent less stable income.")
    notes.append("The debt-to-income ratio is moderate." if applicant["Debt_to_Income_Ratio"] <= 0.35 else "The debt-to-income ratio is relatively high.")
    notes.append("The applicant has no or few existing loans." if applicant["Existing_Loans"] <= 1 else "The applicant has multiple existing loans.")
    return notes

def main():
    parser = argparse.ArgumentParser(description="Predict personal loan eligibility for a new applicant.")
    parser.add_argument("--json", help="Optional JSON object containing applicant fields.")
    args = parser.parse_args()
    if args.json:
        import json
        applicant = json.loads(args.json)
    else:
        applicant = {
            "Age": 30, "Gender": "Male", "Married": "Yes", "Education": "Graduate",
            "Self_Employed": "No", "Applicant_Income": 60000, "Coapplicant_Income": 20000,
            "Loan_Amount": 300000, "Loan_Term": 360, "Credit_History": "Good",
            "Existing_Loans": 1, "Debt_to_Income_Ratio": 0.30,
            "Employment_Status": "Salaried", "Property_Area": "Urban",
        }
    model_name, prediction, probability_map = predict_applicant(applicant)
    print("-" * 40)
    print("PERSONAL LOAN ELIGIBILITY")
    print("-" * 40)
    print(f"Final model: {model_name}")
    print(f"Prediction: {prediction}")
    print(f"Eligibility Probability: {probability_map.get('ELIGIBLE', 0) * 100:.2f}%")
    print(f"Not Eligible Probability: {probability_map.get('NOT_ELIGIBLE', 0) * 100:.2f}%")
    print("\nKey observations:")
    for note in observations(applicant):
        print(f"- {note}")
    print("\nThese probabilities and observations are based on the sample training data. They are not a bank decision or a guaranteed approval.")


main()
