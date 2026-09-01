from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="ML inference API for breast cancer classification",
    version="1.0.0"
)


class PredictionRequest(BaseModel):

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 numerical features required by the model"
    )


@app.get("/")
def home():

    return {
        "message": "Breast Cancer Prediction API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def prediction(request: PredictionRequest):

    try:

        result = predict(request.features)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )