"""Batch prediction: generate forecasts and write to the prediction store.

This module loads a trained LightGBM model, generates predictions for
all items in a store, and writes results to Azure SQL DB.
"""

import json

import pandas as pd

from src.db.connection import get_connection


def write_forecasts(
    predictions: pd.DataFrame,
    model_version: str,
) -> int:
    """Write forecast rows to the Azure SQL DB `forecasts` table.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns: item_id, store_id, forecast_date, predicted_sales.
    model_version : str
        Version tag for this model run (e.g. "v1.0").

    Returns
    -------
    int
        Number of rows inserted.
    """
    conn = get_connection()
    cursor = conn.cursor()

    rows = 0
    for _, row in predictions.iterrows():
        cursor.execute(
            "INSERT INTO forecasts (item_id, store_id, forecast_date, predicted_sales, model_version) "
            "VALUES (?, ?, ?, ?, ?)",
            row["item_id"],
            row["store_id"],
            row["forecast_date"],
            float(row["predicted_sales"]),
            model_version,
        )
        rows += 1

    conn.commit()
    cursor.close()
    conn.close()
    return rows


def log_model_run(
    model_version: str,
    smape: float,
    mae: float,
    horizon_days: int,
    num_items: int,
    store_id: str,
    parameters: dict | None = None,
) -> None:
    """Log a model training run to the `model_runs` table.

    Parameters
    ----------
    model_version : str
        Unique version tag (e.g. "v1.0").
    smape : float
        Symmetric mean absolute percentage error on holdout.
    mae : float
        Mean absolute error on holdout.
    horizon_days : int
        Number of days forecasted.
    num_items : int
        Number of items forecasted.
    store_id : str
        Store identifier (e.g. "CA_1").
    parameters : dict, optional
        Model hyperparameters to log as JSON.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO model_runs "
        "(model_version, smape, mae, horizon_days, num_items, store_id, parameters) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        model_version,
        smape,
        mae,
        horizon_days,
        num_items,
        store_id,
        json.dumps(parameters) if parameters else None,
    )

    conn.commit()
    cursor.close()
    conn.close()
