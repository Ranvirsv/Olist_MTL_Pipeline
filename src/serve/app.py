import sys
import os

# Make src/models importable so MLflow can resolve the mmoe module when unpickling
_models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
if _models_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_models_dir))

import torch
import numpy as np
import joblib
from pydantic import BaseModel, field_validator, Field
from fastapi import FastAPI
import mlflow.pytorch
import mlflow
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
from loguru import logger
from typing import Any
import math

class OrderFeatures(BaseModel):
    customer_city: str = None
    customer_state: str = Field(..., min_length=2, max_length=2)
    price: float = Field(..., ge=0.0)
    freight_value: float = Field(..., ge=0.0)
    product_weight_g: float = Field(..., ge=0.0)
    product_category_name: str = None
    seller_state: str = Field(..., max_length=2)
    total_payment_value: float = Field(..., ge=0.0)
    max_installments: int = Field(..., ge=0)
    product_description_length: float = Field(..., ge=0.0)
    product_photos_qty: float = Field(..., ge=0.0)
    product_volume_cm3: float = Field(..., ge=0.0)
    num_items: int = Field(..., ge=1)
    seller_order_count: int = Field(..., ge=1)
    seller_avg_review: float = Field(..., ge=1.0, le=5.0)
    customer_lat: float = None
    customer_lng: float = None
    seller_lat: float = None
    seller_lng: float = None
    hour_sin: float = None
    hour_cos: float = None
    day_sin: float = None
    day_cos: float = None
    month_sin: float = None
    month_cos: float = None

    @field_validator(
        'price', 'freight_value', 'product_weight_g',
        'total_payment_value', 'max_installments', 'customer_lat', 'customer_lng',
        'seller_lat', 'seller_lng', 'seller_state', 
        'product_category_name', 'customer_city',
        'customer_state', 'product_description_length', 'product_photos_qty',
        'product_volume_cm3', 'num_items', 'seller_order_count', 'seller_avg_review',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        mode='before'
    )
    @classmethod
    def isnan(cls, value: Any):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

class PredictedResponse(BaseModel):
    delivery_days: float
    review_score: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(find_dotenv())
    ip_address = os.getenv("MLFLOW_TRACKING_URI")
    root_dir = os.getenv("PROJECT_ROOT")
    app.state.device = "mps" if torch.backends.mps.is_available() else "cpu"
    try: 
        mlflow.set_tracking_uri(
            f"http://{ip_address}"
        )
        logger.info(f"MLFLOW Tracking URI: {mlflow.get_tracking_uri()}")
        model_name = "MMoE Best Bal_Acc"
        model_version = 1
        model = mlflow.pytorch.load_model(f"models:/{model_name}/{model_version}")
        model.to(app.state.device)
        model.eval()

        app.state.model = model
    except Exception as e:
        logger.error(f"Error loading model: {e}")

    try:
        scaler = joblib.load(f"{root_dir}/data/preprocessed/delivery_scaler.pkl")
        app.state.scaler = scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")

    yield
    
    logger.info("Shutting down server, clearing models from memory.")
    app.state.model = None
    app.state.scaler = None

app = FastAPI(title="Olist Logistics AI", lifespan=lifespan)

@app.post('/predict', response_model=PredictedResponse)
async def predict(order: OrderFeatures) -> PredictedResponse:
    model = app.state.model
    scaler = app.state.scaler
    device = app.state.device
    order_features_numpy = np.array([list(order.model_dump().values())], dtype=np.float32)
    order_features_tensor = torch.from_numpy(order_features_numpy).to(device)
    with torch.no_grad():
        out_delivery, out_satisfaction = model(order_features_tensor)
        out_delivery = scaler.inverse_transform(out_delivery.detach().cpu().numpy().reshape(-1, 1)).flatten()
        out_satisfaction = torch.sigmoid(out_satisfaction).detach().cpu().numpy().reshape(-1, 1).flatten()
        out_satisfaction = (out_satisfaction > 0.5).astype(int)
    return PredictedResponse(delivery_days=out_delivery[0], review_score=out_satisfaction[0])