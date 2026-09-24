"""Synthetic Olist-shaped dataset for tests, local dev, and CI/Docker builds.

Not real training data - never use it to justify a production model's metrics.
"""

import numpy as np
import pandas as pd


def build_synthetic_labeled_dataset(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    purchase = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 270, size=rows), unit="D"
    )
    approved = purchase + pd.to_timedelta(rng.integers(1, 48, size=rows), unit="h")
    estimated = purchase + pd.to_timedelta(rng.integers(5, 30, size=rows), unit="D")
    is_delayed = rng.integers(0, 2, size=rows)

    return pd.DataFrame(
        {
            "order_purchase_timestamp": purchase,
            "order_approved_at": approved,
            "order_estimated_delivery_date": estimated,
            "total_price": rng.uniform(10, 500, size=rows),
            "total_freight": rng.uniform(5, 50, size=rows),
            "items_count": rng.integers(1, 5, size=rows),
            "unique_products": rng.integers(1, 3, size=rows),
            "unique_sellers": rng.integers(1, 2, size=rows),
            "total_payment_value": rng.uniform(10, 550, size=rows),
            "max_payment_installments": rng.integers(1, 10, size=rows),
            "payment_types_count": rng.integers(1, 2, size=rows),
            "customer_state": rng.choice(["SP", "RJ", "MG"], size=rows),
            "primary_payment_type": rng.choice(["credit_card", "boleto"], size=rows),
            "is_delayed": is_delayed,
        }
    )
