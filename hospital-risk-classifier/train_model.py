from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "patient_data.csv"
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "plots"
TARGET = "Risk_Level"
FEATURES = ["Age", "Gender", "Heart_Rate", "Systolic_BP", "Diastolic_BP", "Temperature", "Oxygen_Saturation", "Respiratory_Rate", "BMI", "Previous_Hospitalization", "Chronic_Condition", "Medication_Adherence"]
NUMERIC = ["Age", "Heart_Rate", "Systolic_BP", "Diastolic_BP", "Temperature", "Oxygen_Saturation", "Respiratory_Rate", "BMI"]
CATEGORICAL = ["Gender", "Previous_Hospitalization", "Chronic_Condition", "Medication_Adherence"]
LABELS = ["LOW", "MEDIUM", "HIGH"]

def make_preprocessor(scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([("numeric", Pipeline(numeric_steps), NUMERIC), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)])

def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, predictions, labels=LABELS, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_test, predictions), "precision": p, "recall": r, "f1_score": f, "predictions": predictions}

def save_confusion_matrix(name, y_test, predictions):
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted risk level")
    plt.ylabel("Actual risk level")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close()

def create_eda(df):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, order=LABELS, hue=TARGET, legend=False, ax=ax)
    ax.set(title="Risk-Level Distribution", xlabel="Risk level", ylabel="Number of patients")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "risk_distribution.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Age", "Oxygen_Saturation", "Heart_Rate"]):
        sns.boxplot(data=df, x=TARGET, y=col, order=LABELS, hue=TARGET, legend=False, ax=ax)
        ax.set_title(f"{col} by Risk Level"); ax.set_xlabel("Risk level")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "vital_signs_by_risk.png", dpi=150); plt.close(fig)
    corr = df[NUMERIC + [TARGET]].copy(); corr[TARGET] = corr[TARGET].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    plt.figure(figsize=(9, 7)); sns.heatmap(corr.corr(numeric_only=True), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap"); plt.tight_layout(); plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150); plt.close()

def main():
    MODEL_DIR.mkdir(exist_ok=True); PLOT_DIR.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_dataset.py")
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values before imputation:\n", df.isnull().sum())
    df = df.drop_duplicates().copy()
    create_eda(df)
    x = df[FEATURES]; y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    results = {}
    for k in [3, 5, 7]:
        knn = Pipeline([("preprocessor", make_preprocessor(True)), ("classifier", KNeighborsClassifier(n_neighbors=k))])
        knn.fit(x_train, y_train); result = evaluate(knn, x_test, y_test); results[f"KNN (k={k})"] = {k2: v for k2, v in result.items() if k2 != "predictions"}
        print(f"KNN k={k}: accuracy={result['accuracy']:.3f}, F1={result['f1_score']:.3f}")
    best_k = max([3, 5, 7], key=lambda k: results[f"KNN (k={k})"]["f1_score"])
    knn = Pipeline([("preprocessor", make_preprocessor(True)), ("classifier", KNeighborsClassifier(n_neighbors=best_k))])
    tree = Pipeline([("preprocessor", make_preprocessor(False)), ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))])
    models = {f"KNN (k={best_k})": knn, "Decision Tree": tree}
    for name, model in models.items():
        model.fit(x_train, y_train); result = evaluate(model, x_test, y_test); results[name] = {k: v for k, v in result.items() if k != "predictions"}
        save_confusion_matrix(name, y_test, result["predictions"])
        print(f"\n{name}\n", classification_report(y_test, result["predictions"], labels=LABELS, zero_division=0))
    final_name = max(models, key=lambda name: (results[name]["recall"], results[name]["f1_score"], results[name]["accuracy"]))
    joblib.dump(models[final_name], MODEL_DIR / "final_model.pkl")
    metadata = {"final_model": final_name, "features": FEATURES, "labels": LABELS, "metrics": results}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))
    comparison = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score"]]
    comparison.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), title="Model Comparison")
    plt.ylabel("Score"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(PLOT_DIR / "model_comparison.png", dpi=150); plt.close()
    tree_model = models["Decision Tree"]
    try:
        names = tree_model.named_steps["preprocessor"].get_feature_names_out()
        plt.figure(figsize=(18, 9)); plot_tree(tree_model.named_steps["classifier"], feature_names=names, class_names=LABELS, filled=True, max_depth=3, fontsize=7)
        plt.tight_layout(); plt.savefig(PLOT_DIR / "decision_tree.png", dpi=150); plt.close()
    except Exception as exc: print("Tree plot skipped:", exc)
    print("\nSelected final model:", final_name)
    print("Saved:", MODEL_DIR / "final_model.pkl")

def predict_new_patient(patient: dict):
    model = joblib.load(MODEL_DIR / "final_model.pkl")
    prediction = model.predict(pd.DataFrame([patient]))[0]
    observations = []
    if patient["Oxygen_Saturation"] < 94: observations.append("oxygen saturation is relatively low")
    if patient["Systolic_BP"] >= 140 or patient["Diastolic_BP"] >= 90: observations.append("blood pressure is elevated")
    if patient["Previous_Hospitalization"] == "Yes": observations.append("previous hospitalization is reported")
    if patient["Chronic_Condition"] == "Yes": observations.append("a chronic condition is reported")
    if patient["Medication_Adherence"] == "Poor": observations.append("medication adherence is poor")
    return prediction, observations


main()
