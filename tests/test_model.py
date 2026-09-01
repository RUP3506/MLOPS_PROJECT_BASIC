import joblib
import pandas as pd


def test_model_exists():

    model = joblib.load(
        "models/breast_cancer_model.joblib"
    )

    assert model is not None


def test_model_prediction():

    model = joblib.load(
        "models/breast_cancer_model.joblib"
    )

    X_test = pd.read_csv(
        "data/processed/X_test.csv"
    )

    prediction = model.predict(
        X_test.iloc[[0]]
    )

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]