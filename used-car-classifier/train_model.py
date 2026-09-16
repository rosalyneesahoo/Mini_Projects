from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from preprocessing import FEATURE_COLUMNS, build_preprocessor

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "used_cars.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, pos_label="SUITABLE", zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label="SUITABLE", zero_division=0),
        "F1-Score": f1_score(y_test, predictions, pos_label="SUITABLE", zero_division=0),
    }
    print(f"\n{name} classification report")
    print(classification_report(y_test, predictions, zero_division=0))
    return scores, predictions


def save_confusion_matrix(y_test, predictions, title, filename):
    matrix = confusion_matrix(y_test, predictions, labels=["SUITABLE", "NOT_SUITABLE"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["SUITABLE", "NOT_SUITABLE"], yticklabels=["SUITABLE", "NOT_SUITABLE"])
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=150)
    plt.close()


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print("Initial shape:", df.shape)
    print("Missing values before cleaning:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print("Shape after duplicate removal:", df.shape)
    print("Target distribution:\n", df["Purchase_Suitability"].value_counts())

    X = df[FEATURE_COLUMNS]
    y = df["Purchase_Suitability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    knn_results = []
    knn_models = {}
    for k in [3, 5, 7]:
        model = Pipeline([( "preprocessor", build_preprocessor()), ("classifier", KNeighborsClassifier(n_neighbors=k))])
        model.fit(X_train, y_train)
        scores, predictions = evaluate_model(f"KNN (k={k})", model, X_test, y_test)
        knn_results.append(scores)
        knn_models[k] = (model, predictions)
    best_k = max(knn_results, key=lambda row: row["F1-Score"])["Model"]
    best_k = int(best_k.split("=")[1].rstrip(")"))
    knn_model, knn_predictions = knn_models[best_k]

    tree_model = Pipeline([( "preprocessor", build_preprocessor()), ("classifier", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
    tree_model.fit(X_train, y_train)
    tree_scores, tree_predictions = evaluate_model("Decision Tree", tree_model, X_test, y_test)
    knn_scores = next(row for row in knn_results if row["Model"] == f"KNN (k={best_k})")
    comparison = pd.DataFrame([knn_scores, tree_scores])
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:\n", comparison.to_string(index=False))

    save_confusion_matrix(y_test, knn_predictions, f"KNN (k={best_k}) Confusion Matrix", "knn_confusion_matrix.png")
    save_confusion_matrix(y_test, tree_predictions, "Decision Tree Confusion Matrix", "tree_confusion_matrix.png")

    plt.figure(figsize=(7, 4))
    comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", ylim=(0, 1), rot=0)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    preprocessor = tree_model.named_steps["preprocessor"]
    classifier = tree_model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    importance = pd.Series(classifier.feature_importances_, index=names).sort_values(ascending=False).head(10)
    importance.to_csv(REPORT_DIR / "tree_feature_importance.csv", header=["Importance"])
    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh", color="teal")
    plt.title("Top Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "tree_feature_importance.png", dpi=150)
    plt.close()

    final_model = tree_model if tree_scores["F1-Score"] >= knn_scores["F1-Score"] else knn_model
    selected_name = "Decision Tree" if final_model is tree_model else f"KNN (k={best_k})"
    joblib.dump(final_model, MODEL_DIR / "final_model.pkl")
    joblib.dump({"feature_columns": FEATURE_COLUMNS, "selected_model": selected_name, "random_state": RANDOM_STATE}, MODEL_DIR / "model_metadata.pkl")
    print(f"\nSelected final model: {selected_name}")
    print(f"Saved model to {MODEL_DIR / 'final_model.pkl'}")

    sample = pd.DataFrame([{
        "Age": 4, "Kilometers_Driven": 35000, "Selling_Price": 550000, "Present_Price": 700000,
        "Fuel_Type": "Petrol", "Transmission": "Manual", "Owner_Count": 1, "Engine_CC": 1200,
        "Mileage": 18, "Service_History": "Good", "Accident_History": "No",
    }])
    print("Sample prediction:", final_model.predict(sample)[0])


main()