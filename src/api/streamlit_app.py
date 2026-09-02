import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.predict import predict


FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":microscope:",
)

st.title("Breast Cancer Prediction")
st.write("Enter the 30 measurements required by the trained model.")

with st.form("prediction_form"):
    features = []
    columns = st.columns(3)

    for index, feature_name in enumerate(FEATURE_NAMES):
        with columns[index % 3]:
            features.append(
                st.number_input(
                    feature_name.title(),
                    value=0.0,
                    format="%.6f",
                    key=feature_name,
                )
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        result = predict(features)
        prediction_label = "Malignant" if result["prediction"] == 0 else "Benign"
        st.subheader(prediction_label)
        st.metric("Model confidence", f"{result['probability']:.2%}")
    except Exception as error:
        st.error(f"Prediction failed: {error}")