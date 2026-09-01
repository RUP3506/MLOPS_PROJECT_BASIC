import joblib
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


MODEL_PATH = "models/breast_cancer_model.joblib"


def load_model():

    logger.info("Loading trained model")

    model = joblib.load(MODEL_PATH)

    return model


def predict(features):

    model = load_model()

    input_data = pd.DataFrame(
        [features]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0].max()

    logger.info("Prediction generated successfully")

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }