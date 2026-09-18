from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/customer_data.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
FEATURES = ["Age", "Annual_Income", "Total_Purchases", "Average_Order_Value",
            "Purchase_Frequency", "Total_Spending", "Discount_Usage",
            "Online_Purchases", "Store_Purchases"]
TARGET = "Customer_Class"
RANDOM_STATE = 42

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_dataset.py")
    df = pd.read_csv(DATA_PATH)
    print("First 5 rows:\n", df.head())
    print("\nShape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nData types:\n", df.dtypes)
    print("\nDescriptive statistics:\n", df.describe().T)
    print("\nMissing values:\n", df.isnull().sum())
    print("Duplicate records:", df.duplicated().sum())
    df = df.drop_duplicates().copy()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = []
    for k in (3, 5, 7):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        results.append({"k": k, "accuracy": accuracy_score(y_test, predictions)})
    results_df = pd.DataFrame(results)
    best_k = int(results_df.sort_values("accuracy", ascending=False).iloc[0]["k"])
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    metrics = {
        "selected_k": best_k,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_score_weighted": float(f1),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }
    print("\nK comparison:\n", results_df)
    print("\nSelected K:", best_k)
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    labels = ["LOW_VALUE", "REGULAR", "HIGH_VALUE"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, predictions, labels=labels), annot=True, fmt="d",
                cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("KNN Confusion Matrix")
    plt.xlabel("Predicted class"); plt.ylabel("Actual class")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    metric_values = [metrics["accuracy"], metrics["precision_weighted"], metrics["recall_weighted"], metrics["f1_score_weighted"]]
    plt.figure(figsize=(7, 4)); plt.bar(["Accuracy", "Precision", "Recall", "F1-Score"], metric_values, color="#4472C4")
    plt.ylim(0, 1); plt.ylabel("Score"); plt.title("Selected KNN Model Performance"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4)); plt.plot(results_df["k"], results_df["accuracy"], marker="o")
    plt.xticks([3, 5, 7]); plt.xlabel("K value"); plt.ylabel("Accuracy"); plt.title("K Value vs Accuracy")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "k_vs_accuracy.png", dpi=150); plt.close()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)
    X_all_scaled = scaler.transform(imputer.transform(df[FEATURES]))
    all_pca = pca.transform(X_all_scaled)
    pca_df = pd.DataFrame({"PC1": all_pca[:, 0], "PC2": all_pca[:, 1], "Customer_Class": y.values})
    pca_df.to_csv(OUTPUT_DIR / "pca_components.csv", index=False)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Customer_Class", hue_order=labels, alpha=0.75)
    plt.title("Retail Customers in PCA Space"); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "pca_visualization.png", dpi=150); plt.close()
    explained = pca.explained_variance_ratio_.tolist()
    print("\nPCA explained variance ratio:", explained)
    print("Cumulative explained variance:", sum(explained))

    joblib.dump(model, MODEL_DIR / "knn_model.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(pca, MODEL_DIR / "pca.pkl")
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"k_comparison": results, **metrics, "pca_explained_variance_ratio": explained}, file, indent=2)
    print("\nSaved model artifacts to models/ and charts/metrics to outputs/")


main()
