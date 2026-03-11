from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Annotated, Any
import math

class OlistAnalyticalRow(BaseModel):
    order_id: str
    customer_unique_id: str
    customer_city: str
    customer_state: str = Field(..., min_length=2, max_length=2)
    order_purchase_timestamp: datetime
    
    # Targets
    delivery_days: Optional[float] = Field(None, description="Target A: Days to deliver")
    review_score: Optional[int] = Field(None, ge=1, le=5, description="Target B: 1 to 5 rating")
    review_comment_message: Optional[str] = None 
    
    # Features
    price: Optional[float] = Field(None, ge=0.0)
    freight_value: Optional[float] = Field(None, ge=0.0)
    product_weight_g: Optional[float] = Field(None, ge=0.0)
    product_category_name: Optional[str] = None
    seller_state: Optional[str] = Field(None, max_length=2)
    total_payment_value: Optional[float] = Field(None, ge=0.0)
    max_installments: Optional[int] = Field(None, ge=0)
    customer_lat: Optional[float] = None
    customer_lng: Optional[float] = None
    seller_lat: Optional[float] = None
    seller_lng: Optional[float] = None

    ## add all option fileds to be validated
    @field_validator(
        'delivery_days', 'review_score', 'price', 'freight_value', 'product_weight_g', 
        'total_payment_value', 'max_installments', 'customer_lat', 'customer_lng', 
        'seller_lat', 'seller_lng', 'seller_state', 'review_comment_message', 
        'product_category_name', 'order_purchase_timestamp', 'customer_city', 
        'customer_state', 'order_id', 'customer_unique_id', mode='before'
    )
    @classmethod
    def isnan(cls, value: Any):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    

