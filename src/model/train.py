"""Model training utilities for M5 demand forecasting."""

import numpy as np
import pandas as pd

from src.features import build_features

FEATURE_COLUMNS = [
    "day_num",
    "wday",
    "month",
    "year",
    "is_weekend",
    "has_event",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
    "sales_rolling_std_7",
    "sales_rolling_std_28",
]


def train_lightgbm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
) -> tuple:
    """Train a LightGBM regressor with early stopping.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with feature columns and 'sales'.
    df_val : pd.DataFrame
        Validation data with feature columns and 'sales'.
    feature_cols : list[str], optional
        Features to use. Defaults to :data:`FEATURE_COLUMNS`.
    params : dict, optional
        LightGBM parameters. Merged with sensible defaults.

    Returns
    -------
    tuple[lgb.Booster, dict]
        Trained model and metrics dict with 'mae' and 'rmse' keys.
    """
    import lightgbm as lgb

    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    default_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    if params:
        default_params.update(params)

    train_data = lgb.Dataset(df_train[feature_cols], label=df_train["sales"])
    val_data = lgb.Dataset(df_val[feature_cols], label=df_val["sales"], reference=train_data)

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    y_pred = model.predict(df_val[feature_cols], num_iteration=model.best_iteration)
    y_true = df_val["sales"].values
    errors = y_true - y_pred

    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
    return model, metrics


def prepare_item_data(
    df_store: pd.DataFrame,
    item_id: str,
    horizon: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train/val splits for a single item.

    Filters *df_store* to *item_id*, engineers features via
    :func:`build_features`, drops NaN rows introduced by lags, and
    splits into train and validation sets.

    Parameters
    ----------
    df_store : pd.DataFrame
        Store-level data with all items.
    item_id : str
        Item to filter on.
    horizon : int
        Number of days to hold out for validation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_train, df_val)`` — ready for :func:`train_lightgbm`.
    """
    df_item = df_store[df_store["item_id"] == item_id].copy().sort_values("day_num")
    df_item = build_features(df_item)
    df_item = df_item.dropna(subset=FEATURE_COLUMNS)

    df_train = df_item.iloc[:-horizon]
    df_val = df_item.iloc[-horizon:]
    return df_train, df_val


def train_all_items(
    df_store: pd.DataFrame,
    horizon: int = 28,
    params: dict | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Train per-item models and collect predictions.

    Parameters
    ----------
    df_store : pd.DataFrame
        Store-level data (all items, calendar-enriched).
    horizon : int
        Validation horizon in days.
    params : dict, optional
        LightGBM parameters passed to :func:`train_lightgbm`.

    Returns
    -------
    tuple[pd.DataFrame, float, float]
        ``(predictions_df, overall_mae, overall_rmse)``

        *predictions_df* has columns: item_id, store_id,
        forecast_date, predicted_sales — matching the
        ``write_forecasts`` schema.
    """
    item_ids = df_store["item_id"].unique()
    all_preds: list[pd.DataFrame] = []
    all_maes: list[float] = []
    all_rmses: list[float] = []

    for i, item_id in enumerate(item_ids):
        if i % 100 == 0:
            print(f"Training item {i + 1}/{len(item_ids)}...")
        df_train, df_val = prepare_item_data(df_store, item_id, horizon)

        if len(df_train) == 0 or len(df_val) == 0:
            continue

        model, metrics = train_lightgbm(df_train, df_val, params=params)
        all_maes.append(metrics["mae"])
        all_rmses.append(metrics["rmse"])

        y_pred = model.predict(df_val[FEATURE_COLUMNS], num_iteration=model.best_iteration)

        preds = pd.DataFrame(
            {
                "item_id": item_id,
                "store_id": df_val["store_id"].iloc[0],
                "forecast_date": df_val["cal_date"].values,
                "predicted_sales": y_pred,
            }
        )
        all_preds.append(preds)

    predictions_df = pd.concat(all_preds, ignore_index=True)
    overall_mae = float(np.mean(all_maes))
    overall_rmse = float(np.mean(all_rmses))
    return predictions_df, overall_mae, overall_rmse
