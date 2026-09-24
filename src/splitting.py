import pandas as pd

DEFAULT_TRAIN_END = pd.Timestamp("2018-05-31 23:59:59")
DEFAULT_VAL_END = pd.Timestamp("2018-07-15 23:59:59")


def time_based_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    val_end: pd.Timestamp = DEFAULT_VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    train_df = ordered[ordered["order_purchase_timestamp"] <= train_end].copy()
    val_df = ordered[
        (ordered["order_purchase_timestamp"] > train_end)
        & (ordered["order_purchase_timestamp"] <= val_end)
    ].copy()
    test_df = ordered[ordered["order_purchase_timestamp"] > val_end].copy()
    return train_df, val_df, test_df
