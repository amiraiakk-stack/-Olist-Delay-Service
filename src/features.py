import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "total_price",
    "total_freight",
    "items_count",
    "unique_products",
    "unique_sellers",
    "total_payment_value",
    "max_payment_installments",
    "payment_types_count",
    "purchase_hour",
    "purchase_dayofweek",
    "purchase_is_weekend",
    "purchase_month",
    "estimated_shipping_days",
    "approval_delay_hours",
    "freight_ratio",
]
CATEGORICAL_FEATURES = ["customer_state", "primary_payment_type"]


def _as_datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce")


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "order_purchase_timestamp" in result:
        purchase_time = _as_datetime(result, "order_purchase_timestamp")
        result["purchase_hour"] = purchase_time.dt.hour
        result["purchase_dayofweek"] = purchase_time.dt.dayofweek
        result["purchase_is_weekend"] = result["purchase_dayofweek"].isin([5, 6]).astype(int)
        result["purchase_month"] = purchase_time.dt.month
    if "order_estimated_delivery_date" in result and "order_purchase_timestamp" in result:
        result["estimated_shipping_days"] = (
            _as_datetime(result, "order_estimated_delivery_date")
            - _as_datetime(result, "order_purchase_timestamp")
        ).dt.total_seconds() / (24 * 3600)
    if "order_approved_at" in result and "order_purchase_timestamp" in result:
        result["approval_delay_hours"] = (
            _as_datetime(result, "order_approved_at")
            - _as_datetime(result, "order_purchase_timestamp")
        ).dt.total_seconds() / 3600
    else:
        result["approval_delay_hours"] = np.nan
    if "total_freight" in result and "total_price" in result:
        result["freight_ratio"] = result["total_freight"] / (result["total_price"] + 1e-5)
    result.columns = [str(column) for column in result.columns]
    return result
