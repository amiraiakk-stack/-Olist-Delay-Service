import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

SOURCE_TABLES = {
    "orders": "olist_orders_dataset",
    "items": "olist_order_items_dataset",
    "payments": "olist_order_payments_dataset",
    "reviews": "olist_order_reviews_dataset",
    "customers": "olist_customers_dataset",
    "products": "olist_products_dataset",
    "sellers": "olist_sellers_dataset",
    "geolocation": "olist_geolocation_dataset",
}


def build_engine_from_env() -> Engine:
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASS"]
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{name}")


def read_source_tables(engine: Engine) -> dict[str, pd.DataFrame]:
    return {name: pd.read_sql_table(table, engine) for name, table in SOURCE_TABLES.items()}


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    return (
        items.groupby("order_id")
        .agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            items_count=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            unique_sellers=("seller_id", "nunique"),
        )
        .reset_index()
    )


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    return (
        payments.groupby("order_id")
        .agg(
            total_payment_value=("payment_value", "sum"),
            max_payment_installments=("payment_installments", "max"),
            payment_types_count=("payment_type", "nunique"),
            primary_payment_type=("payment_type", lambda x: x.mode()[0] if not x.empty else None),
        )
        .reset_index()
    )


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    return (
        reviews.groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            has_review_comment=("review_comment_message", lambda x: int(x.notna().any())),
        )
        .reset_index()
    )


def build_ml_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    items_agg = aggregate_items(tables["items"])
    payments_agg = aggregate_payments(tables["payments"])
    reviews_agg = aggregate_reviews(tables["reviews"])

    customer_columns = [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ]
    customers = tables["customers"][customer_columns]
    ml_df = tables["orders"].merge(customers, on="customer_id", how="left")
    ml_df = ml_df.merge(items_agg, on="order_id", how="left")
    ml_df = ml_df.merge(payments_agg, on="order_id", how="left")
    ml_df = ml_df.merge(reviews_agg, on="order_id", how="left")

    if ml_df["order_id"].nunique() != len(ml_df):
        raise ValueError("Duplicate order_id found after merging")
    return ml_df
