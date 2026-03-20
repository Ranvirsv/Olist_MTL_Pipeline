import sys
import os
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, field_validator, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as rt
from src.features.haversine import haversine_distance
from contextlib import asynccontextmanager
from loguru import logger
from typing import Any
import math

##=====================================================================================
##                                  PATH SETUP
##=====================================================================================

# Make src/features importable so CyclicEncoder can be resolved when unpickling the preprocessor
_features_dir = os.path.join(os.path.dirname(__file__), "..", "features")
if _features_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_features_dir))

##=====================================================================================
##                                  PYDANTIC MODELS
##=====================================================================================

class OrderFeatures(BaseModel):
    customer_city: str = None
    customer_state: str = Field(..., min_length=2, max_length=2)
    price: float = Field(..., ge=0.0)
    total_freight_value: float = Field(..., ge=0.0)
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
    order_purchase_timestamp: str

    @field_validator(
        'price', 'total_freight_value', 'product_weight_g',
        'total_payment_value', 'max_installments', 'customer_lat', 'customer_lng',
        'seller_lat', 'seller_lng', 'seller_state', 
        'product_category_name', 'customer_city',
        'customer_state', 'product_description_length', 'product_photos_qty',
        'product_volume_cm3', 'num_items', 'seller_order_count', 'seller_avg_review',
        'order_purchase_timestamp',
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

##=====================================================================================
##                                FAST API SETUP
##=====================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup Device for ONNX (CPU is standard for free cloud tiers)
    app.state.device = "cpu" 
    
    # Dynamically find the root of your project
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    inference_dir = os.path.join(base_dir, "src", "models", "inference")

    try:
        # 1. Load ONNX Model
        model_path = os.path.join(inference_dir, "mmoe_production.onnx")
        app.state.ort_session = rt.InferenceSession(model_path)
        
        # 2. Load Feature Scaler (Pipeline)
        feature_scaler_path = os.path.join(inference_dir, "feature_preprocessor.pkl")
        app.state.feature_scaler = joblib.load(feature_scaler_path)
        
        # 3. Load Target Scaler
        target_scaler_path = os.path.join(inference_dir, "delivery_scaler.pkl")
        app.state.target_scaler = joblib.load(target_scaler_path)
        
        logger.info("ONNX Model and Scalers loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading inference assets: {e}")

    yield
    
    logger.info("Shutting down server, clearing models from memory.")
    app.state.ort_session = None
    app.state.feature_scaler = None
    app.state.target_scaler = None

app = FastAPI(title="Olist Logistics AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

##=====================================================================================
##                                  PREDICTION ENDPOINT
##=====================================================================================

@app.post('/predict', response_model=PredictedResponse)
async def predict(order: OrderFeatures) -> PredictedResponse:
    df_input = pd.DataFrame([order.model_dump()])
    df_input['geo_distance_km'] = haversine_distance(
        df_input['customer_lat'], df_input['customer_lng'],
        df_input['seller_lat'], df_input['seller_lng']
    )
    scaled_features = app.state.feature_scaler.transform(df_input).astype(np.float32)

    out_delivery, out_satisfaction = app.state.ort_session.run(
        None, {"input_features": scaled_features}
    )

    delivery_days = app.state.target_scaler.inverse_transform(
        out_delivery.reshape(-1, 1)
    ).flatten()[0]
    review_score = int(sigmoid(out_satisfaction).flatten()[0] > 0.5)

    return PredictedResponse(delivery_days=delivery_days, review_score=review_score)