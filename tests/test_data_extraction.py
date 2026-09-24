import pandas as pd
import pytest

from src.data_extraction import (
    aggregate_items,
    aggregate_payments,
    aggregate_reviews,
    build_ml_dataset,
)


def test_aggregate_items_sums_and_counts_per_order():
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "price": [10.0, 20.0, 5.0],
            "freight_value": [1.0, 2.0, 0.5],
            "product_id": ["p1", "p2", "p1"],
            "seller_id": ["s1", "s1", "s2"],
        }
    )
    result = aggregate_items(items).set_index("order_id")

    assert result.loc["o1", "total_price"] == 30.0
    assert result.loc["o1", "total_freight"] == 3.0
    assert result.loc["o1", "items_count"] == 2
    assert result.loc["o1", "unique_products"] == 2
    assert result.loc["o1", "unique_sellers"] == 1
    assert result.loc["o2", "total_price"] == 5.0


def test_aggregate_payments_finds_max_installments_and_primary_type():
    payments = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o1"],
            "payment_value": [50.0, 30.0, 20.0],
            "payment_installments": [1, 3, 1],
            "payment_type": ["credit_card", "credit_card", "boleto"],
        }
    )
    result = aggregate_payments(payments).set_index("order_id")

    assert result.loc["o1", "total_payment_value"] == 100.0
    assert result.loc["o1", "max_payment_installments"] == 3
    assert result.loc["o1", "payment_types_count"] == 2
    assert result.loc["o1", "primary_payment_type"] == "credit_card"


def test_aggregate_reviews_averages_score_and_flags_comments():
    reviews = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "review_score": [4, 2, 5],
            "review_comment_message": [None, "great!", None],
        }
    )
    result = aggregate_reviews(reviews).set_index("order_id")

    assert result.loc["o1", "review_score"] == 3.0
    assert result.loc["o1", "has_review_comment"] == 1
    assert result.loc["o2", "has_review_comment"] == 0


def test_build_ml_dataset_merges_all_tables_one_row_per_order():
    tables = {
        "orders": pd.DataFrame({"order_id": ["o1", "o2"], "customer_id": ["c1", "c2"]}),
        "customers": pd.DataFrame(
            {
                "customer_id": ["c1", "c2"],
                "customer_unique_id": ["u1", "u2"],
                "customer_zip_code_prefix": ["1000", "2000"],
                "customer_city": ["sao paulo", "rio"],
                "customer_state": ["SP", "RJ"],
            }
        ),
        "items": pd.DataFrame(
            {
                "order_id": ["o1", "o2"],
                "order_item_id": [1, 1],
                "price": [10.0, 20.0],
                "freight_value": [1.0, 2.0],
                "product_id": ["p1", "p2"],
                "seller_id": ["s1", "s2"],
            }
        ),
        "payments": pd.DataFrame(
            {
                "order_id": ["o1", "o2"],
                "payment_value": [11.0, 22.0],
                "payment_installments": [1, 2],
                "payment_type": ["credit_card", "boleto"],
            }
        ),
        "reviews": pd.DataFrame(
            {
                "order_id": ["o1", "o2"],
                "review_score": [5, 3],
                "review_comment_message": [None, None],
            }
        ),
    }

    result = build_ml_dataset(tables)

    assert len(result) == 2
    assert result["order_id"].nunique() == len(result)
    assert set(result["customer_state"]) == {"SP", "RJ"}
    assert result.set_index("order_id").loc["o1", "total_price"] == 10.0


def test_build_ml_dataset_rejects_duplicate_order_ids():
    tables = {
        "orders": pd.DataFrame({"order_id": ["o1", "o1"], "customer_id": ["c1", "c1"]}),
        "customers": pd.DataFrame(
            {
                "customer_id": ["c1"],
                "customer_unique_id": ["u1"],
                "customer_zip_code_prefix": ["1000"],
                "customer_city": ["sao paulo"],
                "customer_state": ["SP"],
            }
        ),
        "items": pd.DataFrame(
            {
                "order_id": [],
                "order_item_id": [],
                "price": [],
                "freight_value": [],
                "product_id": [],
                "seller_id": [],
            }
        ),
        "payments": pd.DataFrame(
            {"order_id": [], "payment_value": [], "payment_installments": [], "payment_type": []}
        ),
        "reviews": pd.DataFrame({"order_id": [], "review_score": [], "review_comment_message": []}),
    }

    with pytest.raises(ValueError):
        build_ml_dataset(tables)
