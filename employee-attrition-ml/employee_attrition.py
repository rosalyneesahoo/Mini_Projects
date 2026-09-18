
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
FEATURES = [
    "Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany",
    "OverTime", "JobLevel", "WorkLifeBalance", "JobInvolvement",
    "YearsSinceLastPromotion", "TrainingTimesLastYear", "TotalWorkingYears",
    "EnvironmentSatisfaction", "DistanceFromHome", "BusinessTravel",
]
TARGET = "Attrition"
NUMERIC_FEATURES = [
    "Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "JobLevel",
    "WorkLifeBalance", "JobInvolvement", "YearsSinceLastPromotion",
    "TrainingTimesLastYear", "TotalWorkingYears", "EnvironmentSatisfaction",
    "DistanceFromHome",
]
CATEGORICAL_FEATURES = ["OverTime", "BusinessTravel"]


def load_and_inspect_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    print("\nDATASET INSPECTION")
    print("First five rows:\n", df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumn names:\n", list(df.columns))
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nDescriptive statistics:\n", df.describe(include="all").transpose().head(20))
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates().copy()
        print(f"Removed {duplicate_count} duplicate rows.")
    else:
        print("No duplicate rows were found.")
    return df


def make_eda_plots(df):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, order=["No", "Yes"], ax=ax)
    ax.set_title("Employee Attrition Distribution")
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Number of Employees")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_attrition_distribution.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="OverTime", hue=TARGET, order=["No", "Yes"], ax=ax)
    ax.set_title("Overtime and Attrition")
    ax.set_xlabel("Works Overtime")
    ax.set_ylabel("Number of Employees")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_overtime_attrition.png", dpi=150)
    plt.close(fig)
    for number, feature, title, xlabel in [
        (3, "JobSatisfaction", "Job Satisfaction and Attrition", "Job Satisfaction (1-4)"),
        (4, "MonthlyIncome", "Monthly Income and Attrition", "Monthly Income"),
        (5, "YearsAtCompany", "Years at Company and Attrition", "Years at Company"),
        (6, "WorkLifeBalance", "Work-Life Balance and Attrition", "Work-Life Balance (1-4)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x=TARGET, y=feature, order=["No", "Yes"], ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Attrition")
        ax.set_ylabel(xlabel)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"0{number}_{feature.lower()}_attrition.png", dpi=150)
        plt.close(fig)
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Variables")
    ax.set_xlabel("Numerical Features")
    ax.set_ylabel("Numerical Features")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nEDA graphs saved to: {OUTPUT_DIR}")

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

def choose_knn_k(X_train, y_train):
    scores = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for k in [3, 5, 7]:
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores[k] = float(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean())
    selected_k = max(scores, key=scores.get)
    print("\nKNN training-only cross-validation F1 scores:", scores)
    print("Selected K:", selected_k)
    return selected_k, scores

def evaluate_model(name, model, X_test, y_test, results):
    predictions = model.predict(X_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
    }
    results.append(metrics)
    print(f"\n{name}")
    print(classification_report(y_test, predictions, target_names=["Stay (No)", "Leave (Yes)"], zero_division=0))
    print("Confusion matrix [actual rows: Stay, Leave; predicted columns: Stay, Leave]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    return predictions

def save_confusion_matrix(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted: Stay (0) / Leave (1)")
    ax.set_ylabel("Actual: Stay (0) / Leave (1)")
    fig.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    fig.savefig(OUTPUT_DIR / f"confusion_{safe_name}.png", dpi=150)
    plt.close(fig)

def main():
    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING")
    print("=" * 60)
    df = load_and_inspect_data()
    make_eda_plots(df)
    X = df[FEATURES].copy()
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isnull().any():
        raise ValueError("Target contains values other than No and Yes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("Target encoding: No = 0 (stay), Yes = 1 (leave)")

    selected_k, _ = choose_knn_k(X_train, y_train)
    model_definitions = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=selected_k),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    }
    models = {}
    for name, estimator in model_definitions.items():
        models[name] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        models[name].fit(X_train, y_train)
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    results = []
    for name, model in models.items():
        evaluate_model(name, model, X_test, y_test, results)
        save_confusion_matrix(name, model, X_test, y_test)
    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_model_comparison.png", dpi=150)
    plt.close(fig)
    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    print(f"\nSelected final model by highest test F1-score: {best_name}")
    tree = models["Decision Tree"]
    preprocessor = tree.named_steps["preprocessor"]
    tree_estimator = tree.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()
    importances = pd.DataFrame({
        "Feature": transformed_names,
        "Importance": tree_estimator.feature_importances_,
    }).sort_values("Importance", ascending=False)
    importances.to_csv(OUTPUT_DIR / "decision_tree_feature_importance.csv", index=False)
    print("\nDECISION TREE FEATURE IMPORTANCE")
    print(importances.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    top_importances = importances.head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_importances["Feature"], top_importances["Importance"])
    ax.set_title("Decision Tree Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_decision_tree_feature_importance.png", dpi=150)
    plt.close(fig)
    sample_employee = pd.DataFrame([{
        "Age": 30, "MonthlyIncome": 4500, "JobSatisfaction": 2,
        "YearsAtCompany": 3, "OverTime": "Yes", "JobLevel": 2,
        "WorkLifeBalance": 2, "JobInvolvement": 2,
        "YearsSinceLastPromotion": 2, "TrainingTimesLastYear": 2,
        "TotalWorkingYears": 7, "EnvironmentSatisfaction": 2,
        "DistanceFromHome": 15, "BusinessTravel": "Travel_Rarely",
    }])
    sample_prediction = int(best_model.predict(sample_employee)[0])
    probabilities = best_model.predict_proba(sample_employee)[0]
    print("\n" + "=" * 60)
    print("SAMPLE EMPLOYEE PREDICTION")
    print("=" * 60)
    print("Prediction:", "LIKELY TO LEAVE" if sample_prediction == 1 else "LIKELY TO STAY")
    print(f"Probability of attrition: {probabilities[1] * 100:.2f}%")
    print(f"Probability of staying: {probabilities[0] * 100:.2f}%")
    print("These are samples of model-estimated probabilities, not guarantees.")

main()
