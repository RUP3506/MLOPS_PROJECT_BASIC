from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_params():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    return params


def train_model():

    logger.info("Starting model training")

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    params = load_params()

    C = params["model"]["C"]
    max_iter = params["model"]["max_iter"]
    random_state = params["model"]["random_state"]

    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state
            )
        )
    ])

    mlflow.set_experiment("breast-cancer-classification")

    with mlflow.start_run():

        model_pipeline.fit(X_train, y_train)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)

        mlflow.sklearn.log_model(
            model_pipeline,
            "model"
        )

        logger.info("Model trained successfully")
        logger.info("Model logged to MLflow")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "breast_cancer_model.joblib"

    joblib.dump(model_pipeline, model_path)

    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    train_model()