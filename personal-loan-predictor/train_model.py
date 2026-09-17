from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from generate_dataset import generate_dataset


RANDOM_STATE = 42
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = ROOT / "plots"
NUMERIC_FEATURES = [
    "Age", "Applicant_Income", "Coapplicant_Income", "Loan_Amount", "Loan_Term",
    "Existing_Loans", "Debt_to_Income_Ratio"
]
CATEGORICAL_FEATURES = [
    "Gender", "Married", "Education", "Self_Employed", "Credit_History",
    "Employment_Status", "Property_Area"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Loan_Eligibility"

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])

def build_pipeline(classifier):
    return Pipeline([("preprocessor", make_preprocessor()), ("classifier", classifier)])

def evaluate_model(name, model, x_test, y_test):
    predictions = model.predict(x_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, pos_label="ELIGIBLE", zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label="ELIGIBLE", zero_division=0),
        "F1-Score": f1_score(y_test, predictions, pos_label="ELIGIBLE", zero_division=0),
    }
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["ELIGIBLE", "NOT_ELIGIBLE"])
    print(f"\n{name}\n{report}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(matrix, index=["ELIGIBLE", "NOT_ELIGIBLE"], columns=["ELIGIBLE", "NOT_ELIGIBLE"]))
    return metrics, matrix

def save_eda_plots(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, order=["ELIGIBLE", "NOT_ELIGIBLE"])
    plt.title("Loan Eligibility Distribution")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "eligibility_distribution.png", dpi=160); plt.close()

    for column, filename, title in [
        ("Applicant_Income", "income_vs_eligibility.png", "Applicant Income by Eligibility"),
        ("Loan_Amount", "loan_amount_vs_eligibility.png", "Loan Amount by Eligibility"),
        ("Debt_to_Income_Ratio", "debt_ratio_vs_eligibility.png", "Debt-to-Income Ratio by Eligibility"),
    ]:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x=TARGET, y=column, order=["ELIGIBLE", "NOT_ELIGIBLE"])
        plt.title(title); plt.tight_layout(); plt.savefig(PLOT_DIR / filename, dpi=160); plt.close()

    credit_table = pd.crosstab(df["Credit_History"], df[TARGET], normalize="index").reindex(columns=["ELIGIBLE", "NOT_ELIGIBLE"])
    credit_table.plot(kind="bar", stacked=True, figsize=(7, 4), color=["#2a9d8f", "#e76f51"])
    plt.title("Credit History vs Eligibility"); plt.ylabel("Proportion"); plt.tight_layout()
    plt.savefig(PLOT_DIR / "credit_history_vs_eligibility.png", dpi=160); plt.close()

    numeric = df.select_dtypes(include=np.number)
    plt.figure(figsize=(9, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Numerical Feature Correlation Heatmap"); plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=160); plt.close()

def main():
    MODEL_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True); PLOT_DIR.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        generate_dataset().to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    print("First five rows:"); print(df.head().to_string(index=False))
    print(f"\nDataset shape: {df.shape}")
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print(f"\nDuplicate records removed: {duplicate_count}")
    print("\nDescriptive statistics:\n", df.describe(include="all").transpose().to_string())
    print("\nEligibility distribution:\n", df[TARGET].value_counts())
    save_eda_plots(df)

    x = df[FEATURES]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Naive Bayes": build_pipeline(GaussianNB()),
        "KNN (k=3)": build_pipeline(KNeighborsClassifier(n_neighbors=3)),
        "KNN (k=5)": build_pipeline(KNeighborsClassifier(n_neighbors=5)),
        "KNN (k=7)": build_pipeline(KNeighborsClassifier(n_neighbors=7)),
        "Decision Tree": build_pipeline(DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
    }
    results, matrices = [], {}
    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics, matrix = evaluate_model(name, model, x_test, y_test)
        results.append(metrics); matrices[name] = matrix; trained_models[name] = model

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as report_file:
        for name, model in trained_models.items():
            report_file.write(f"{name}\n")
            report_file.write(classification_report(y_test, model.predict(x_test), zero_division=0))
            report_file.write("\n\n")

    selection_df = result_df.sort_values(["F1-Score", "Accuracy"], ascending=False)
    final_name = selection_df.iloc[0]["Model"]
    final_model = trained_models[final_name]
    joblib.dump({
        "model": final_model,
        "model_name": final_name,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_classes": ["ELIGIBLE", "NOT_ELIGIBLE"],
    }, MODEL_DIR / "final_model.pkl")

    family_rows = [result_df[result_df["Model"] == "Naive Bayes"].iloc[0]]
    family_rows.append(result_df[result_df["Model"].isin(["KNN (k=3)", "KNN (k=5)", "KNN (k=7)"])].sort_values("F1-Score", ascending=False).iloc[0])
    family_rows.append(result_df[result_df["Model"] == "Decision Tree"].iloc[0])
    family_df = pd.DataFrame(family_rows).reset_index(drop=True)
    family_df.to_csv(OUTPUT_DIR / "model_family_comparison.csv", index=False)
    family_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0)
    plt.title("Model Performance Comparison"); plt.ylabel("Score"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "model_comparison.png", dpi=160); plt.close()

    tree_model = trained_models["Decision Tree"]
    feature_names = tree_model.named_steps["preprocessor"].get_feature_names_out()
    plt.figure(figsize=(22, 12))
    plot_tree(tree_model.named_steps["classifier"], feature_names=feature_names, class_names=["ELIGIBLE", "NOT_ELIGIBLE"], filled=True, rounded=True, max_depth=3, fontsize=7)
    plt.title("Decision Tree (visualized to depth 3)"); plt.tight_layout(); plt.savefig(PLOT_DIR / "decision_tree.png", dpi=160); plt.close()
    importance = pd.DataFrame({"Feature": feature_names, "Importance": tree_model.named_steps["classifier"].feature_importances_}).sort_values("Importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "decision_tree_feature_importance.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE, "rows": int(len(df)), "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
        "selected_model": final_name, "selection_rule": "highest test-set F1-Score, then Accuracy",
        "metrics": result_df.to_dict(orient="records"), "missing_values_before_imputation": df.isna().sum().to_dict(),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    print(f"\nSelected final model: {final_name}")
    print(f"Saved model to: {MODEL_DIR / 'final_model.pkl'}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


main()
