from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_data():
    logger.info("Loading Breast Cancer dataset")

    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(
        dataset.target,
        name="target"
    )

    df = X.copy()
    df["target"] = y

    logger.info("Dataset loaded successfully")
    logger.info("Dataset shape: %s", df.shape)

    return df


def save_raw_data():

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "breast_cancer.csv"

    df = load_data()
    df.to_csv(output_path, index=False)

    logger.info("Raw data saved to %s", output_path)


if __name__ == "__main__":
    save_raw_data()