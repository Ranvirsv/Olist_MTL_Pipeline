from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Annotated, Any
import math

class OlistAnalyticalRaw(BaseModel):
    order_id: str
    customer_unique_id: str
    customer_city: str
    customer_state: str = Field(..., min_length=2, max_length=2)
    order_purchase_timestamp: datetime

    # Targets
    days_to_deliver: Optional[float] = Field(None, description="days to deliver")
    review_score: Optional[int] = Field(None, ge=1, le=5)
    review_comment_message: Optional[str] = None

    # Features
    price: Optional[float] = Field(None, ge=0.0)
    freight_value: Optional[float] = Field(None, ge=0.0)
    product_weight_g: Optional[float] = Field(None, ge=0.0)
    product_category_name: Optional[str] = None

    @field_validator(
        'days_to_deliver', 'price', 'freight_value', 'product_weight_g', 
        'review_comment_message', 'product_category_name', mode='before'
    )
    @classmethod
    def isnan(cls, value: Any):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    

