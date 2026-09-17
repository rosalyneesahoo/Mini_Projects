from pathlib import Path
from generate_dataset import main as generate_main
from src.train_models import evaluate_models
from src.pca_similarity import run_pca
from src.prediction import predict_sample


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "loan_applications.csv"
    artifact_dir = root / "artifacts"
    print("[1/4] Generating synthetic data")
    generate_main()
    print("[2/4] Training and evaluating models")
    evaluate_models(data_path, artifact_dir)
    print("[3/4] Running PCA and similarity exploration")
    run_pca(data_path, artifact_dir)
    print("[4/4] Running sample prediction")
    predict_sample(artifact_dir)
    print(f"Pipeline complete. Outputs are in {artifact_dir}")


main()
