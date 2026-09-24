import pandas as pd

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in DATE_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column])
    return result


def add_delay_label(df: pd.DataFrame) -> pd.DataFrame:
    """Keep delivered orders only and label by comparing actual vs. estimated delivery date."""
    delivered_df = df[df["order_status"] == "delivered"].copy()
    actual = delivered_df["order_delivered_customer_date"]
    estimated = delivered_df["order_estimated_delivery_date"]
    delivered_df["delay_days"] = (actual - estimated).dt.total_seconds() / (24 * 3600)
    delivered_df["is_delayed"] = (delivered_df["delay_days"] > 0).astype(int)
    return delivered_df
