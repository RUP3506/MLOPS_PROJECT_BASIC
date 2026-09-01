import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


def evaluate_model():

    logger.info("Starting model evaluation")

    model_path = "models/breast_cancer_model.joblib"

    model = joblib.load(model_path)

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)

    with open("metrics/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Evaluation completed")
    logger.info("Metrics saved to metrics/metrics.json")


if __name__ == "__main__":
    evaluate_model()