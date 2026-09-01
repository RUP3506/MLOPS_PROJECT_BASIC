from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_params():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    return params


def preprocess_data():

    logger.info("Starting preprocessing")

    input_path = Path("data/raw/breast_cancer.csv")

    df = pd.read_csv(input_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    params = load_params()

    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    logger.info("Preprocessing completed")
    logger.info("Training samples: %s", X_train.shape)
    logger.info("Testing samples: %s", X_test.shape)


if __name__ == "__main__":
    preprocess_data()