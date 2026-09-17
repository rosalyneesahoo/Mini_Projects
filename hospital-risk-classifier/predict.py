from train_model import predict_new_patient

patient = {
    "Age": 62, "Gender": "Female", "Heart_Rate": 96, "Systolic_BP": 150,
    "Diastolic_BP": 92, "Temperature": 37.4, "Oxygen_Saturation": 93,
    "Respiratory_Rate": 22, "BMI": 28, "Previous_Hospitalization": "Yes",
    "Chronic_Condition": "Yes", "Medication_Adherence": "Poor",
}

try:
    prediction, observations = predict_new_patient(patient)
except FileNotFoundError:
    raise SystemExit("Model not found. Run these commands first: python generate_dataset.py && python train_model.py")

print("--------------------------------")
print("PATIENT RISK CLASSIFICATION")
print("--------------------------------")
print(f"Predicted Risk Level: {prediction}")
print("\nObservations from entered data:")
if observations:
    for item in observations:
        print(f"- {item.capitalize()}.")
else:
    print("- No simple threshold-based observations were recorded.")
print("\nThese characteristics were associated with the model's predicted category; they do not establish medical causation.")
print("Academic prototype only. Not a diagnosis or a substitute for healthcare professionals.")
