from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import load_data, make_preprocessor


def top_5_similar(applicant, reference_matrix, reference_frame=None):
    """Return the five most similar rows by cosine similarity, excluding an exact self-match when possible."""
    applicant = np.asarray(applicant).reshape(1, -1)
    scores = cosine_similarity(applicant, reference_matrix).ravel()
    order = np.argsort(-scores)
    results = []
    for index in order:
        results.append({"index": int(index), "similarity": float(scores[index])})
        if len(results) == 5:
            break
    output = pd.DataFrame(results)
    if reference_frame is not None:
        output = output.merge(reference_frame.reset_index().rename(columns={"index": "index"}), on="index", how="left")
    return output

def run_pca(data_path: str | Path, artifact_dir: str | Path, random_state: int = 42):
    artifact_dir, plots_dir = Path(artifact_dir), Path(artifact_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_path)
    reference_frame = X[["credit_score", "annual_income", "loan_amount"]].copy()
    reference_frame["loan_approved"] = y.to_numpy()
    preprocessor = make_preprocessor()
    X_t = preprocessor.fit_transform(X)
    n_components = min(10, X_t.shape[0], X_t.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_t)
    variance = pd.DataFrame({"component": range(1, n_components + 1), "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(artifact_dir / "pca_explained_variance.csv", index=False)
    plt.figure(figsize=(8, 4.5)); plt.plot(variance.component, variance.cumulative_variance, marker="o", color="#2563eb")
    plt.axhline(.80, color="#64748b", linestyle="--", linewidth=1); plt.ylim(0, 1.05)
    plt.xlabel("Number of components"); plt.ylabel("Cumulative explained variance"); plt.title("PCA explained variance")
    plt.tight_layout(); plt.savefig(plots_dir / "pca_explained_variance.png", dpi=170); plt.close()
    if n_components >= 2:
        pca_frame = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "loan_approved": y.astype(str)})
        plt.figure(figsize=(7, 5)); sns.scatterplot(data=pca_frame, x="PC1", y="PC2", hue="loan_approved", palette="Set1", alpha=.7)
        plt.title("Applicants projected onto first two principal components"); plt.tight_layout(); plt.savefig(plots_dir / "pca_scatter.png", dpi=170); plt.close()
    joblib.dump({"preprocessor": preprocessor, "pca": pca}, artifact_dir / "pca_bundle.joblib")
    # Store a simple demonstration: applicant 0 and its closest five rows.
    top_5_similar(X_t[0], X_t, reference_frame).to_csv(artifact_dir / "sample_similar_applicants.csv", index=False)
    print(f"PCA: first 3 components explain {variance.cumulative_variance.iloc[min(2, len(variance)-1)]:.1%}; saved plots to {plots_dir}")
    return variance

def main():
    root = Path(__file__).resolve().parents[1]
    run_pca(root / "data" / "loan_applications.csv", root / "artifacts")


main()
