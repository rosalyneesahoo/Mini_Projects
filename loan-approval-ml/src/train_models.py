from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, feature_names, load_data, make_preprocessor


def evaluate_models(data_path: str | Path, artifact_dir: str | Path, random_state: int = 42):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifact_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=random_state, stratify=y)
    preprocessor = make_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    names = feature_names(preprocessor)
    models = {
        "GaussianNB": GaussianNB(),
        "KNN_k3": KNeighborsClassifier(n_neighbors=3),
        "KNN_k5": KNeighborsClassifier(n_neighbors=5),
        "KNN_k7": KNeighborsClassifier(n_neighbors=7),
        "KNN_k9": KNeighborsClassifier(n_neighbors=9),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state),
    }
    metrics_rows, fitted_models, report_blocks = [], {}, []
    for name, estimator in models.items():
        model = clone(estimator).fit(X_train_t, y_train)
        predictions = model.predict(X_test_t)
        row = {"model": name,
               "accuracy": accuracy_score(y_test, predictions),
               "precision": precision_score(y_test, predictions, zero_division=0),
               "recall": recall_score(y_test, predictions, zero_division=0),
               "f1": f1_score(y_test, predictions, zero_division=0)}
        metrics_rows.append(row)
        fitted_models[name] = model
        report_blocks.append(f"\n{'=' * 72}\n{name}\n{'=' * 72}\n" + classification_report(y_test, predictions, zero_division=0))
        plt.figure(figsize=(4.5, 3.8))
        sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name}: confusion matrix")
        plt.xlabel("Predicted label"); plt.ylabel("Actual label")
        plt.tight_layout(); plt.savefig(plots_dir / f"confusion_{name}.png", dpi=150); plt.close()

    metrics = pd.DataFrame(metrics_rows).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    metrics.to_csv(artifact_dir / "model_metrics.csv", index=False)
    (artifact_dir / "classification_reports.txt").write_text("\n".join(report_blocks), encoding="utf-8")
    long_metrics = metrics.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"], var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5.5)); sns.barplot(data=long_metrics, x="model", y="score", hue="metric")
    plt.ylim(0, 1); plt.xticks(rotation=28, ha="right"); plt.title("Loan approval model comparison")
    plt.tight_layout(); plt.savefig(plots_dir / "model_comparison.png", dpi=170); plt.close()

    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_name]
    joblib.dump(best_model, artifact_dir / "selected_model.joblib")
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = abs(best_model.coef_[0])
    if importances is not None:
        importance_frame = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        importance_frame.to_csv(artifact_dir / "feature_importance.csv", index=False)
        top = importance_frame.head(15).sort_values("importance")
        plt.figure(figsize=(8, 6)); plt.barh(top["feature"], top["importance"], color="#2563eb")
        plt.title(f"Top feature importance: {best_name}"); plt.xlabel("Absolute coefficient / impurity importance")
        plt.tight_layout(); plt.savefig(plots_dir / "feature_importance.png", dpi=170); plt.close()
    metadata = {"selected_model": best_name, "selection_rule": "highest F1, then accuracy",
                "feature_columns": FEATURE_COLUMNS, "target_column": TARGET_COLUMN,
                "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                "best_metrics": {k: float(v) for k, v in metrics.iloc[0].to_dict().items() if k != "model"}}
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected {best_name}: F1={metrics.iloc[0]['f1']:.3f}, accuracy={metrics.iloc[0]['accuracy']:.3f}")
    return metrics, metadata

def main():
    root = Path(__file__).resolve().parents[1]
    evaluate_models(root / "data" / "loan_applications.csv", root / "artifacts")


main()
