import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# shared example request, also used by app/cli.py and Swagger's "Try it out"
_EXAMPLE_REQUEST_PATH = Path(__file__).resolve().parents[1] / "request.example.json"
_EXAMPLE_REQUEST: dict[str, Any] = json.loads(_EXAMPLE_REQUEST_PATH.read_text(encoding="utf-8"))


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE_REQUEST})

    # used to join a logged prediction back to its order later
    order_id: str = Field(min_length=1)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    items_count: int = Field(ge=0)
    unique_products: int = Field(ge=0)
    unique_sellers: int = Field(ge=0)
    total_payment_value: float = Field(ge=0)
    max_payment_installments: int = Field(ge=0)
    payment_types_count: int = Field(ge=0)
    customer_state: str
    primary_payment_type: str
    order_purchase_timestamp: str
    order_estimated_delivery_date: str
    order_approved_at: str | None = None


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"requests": [_EXAMPLE_REQUEST]}})

    requests: list[PredictionRequest] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    threshold: float
    features: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def request_to_payload(request: PredictionRequest) -> dict[str, Any]:
    return request.model_dump()
