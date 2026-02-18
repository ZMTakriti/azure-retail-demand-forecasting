"""Batch prediction: generate forecasts and write to the prediction store.

This module loads a trained LightGBM model, generates predictions for
all items in a store, and writes results to Azure SQL DB.
"""

import json

import pandas as pd

from src.db.connection import get_jdbc_properties, get_jdbc_url


def write_forecasts(
    predictions: pd.DataFrame,
    model_version: str,
) -> int:
    """Write forecast rows to the Azure SQL DB `forecasts` table via Spark JDBC.

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
    from pyspark.sql import SparkSession  # noqa: PLC0415

    df = predictions.copy()
    df["model_version"] = model_version

    spark = SparkSession.getActiveSession()
    spark_df = spark.createDataFrame(df)

    (
        spark_df.write.format("jdbc")
        .option("url", get_jdbc_url())
        .option("dbtable", "forecasts")
        .options(**get_jdbc_properties())
        .mode("append")
        .save()
    )

    return len(df)


def log_model_run(
    model_version: str,
    mae: float,
    rmse: float,
    horizon_days: int,
    num_items: int,
    store_id: str,
    parameters: dict | None = None,
) -> None:
    """Log a model training run to the `model_runs` table via Spark JDBC.

    Parameters
    ----------
    model_version : str
        Unique version tag (e.g. "v1.0").
    mae : float
        Mean absolute error on holdout.
    rmse : float
        Root mean squared error on holdout.
    horizon_days : int
        Number of days forecasted.
    num_items : int
        Number of items forecasted.
    store_id : str
        Store identifier (e.g. "CA_1").
    parameters : dict, optional
        Model hyperparameters to log as JSON.
    """
    from pyspark.sql import SparkSession  # noqa: PLC0415

    df = pd.DataFrame(
        [
            {
                "model_version": model_version,
                "mae": mae,
                "rmse": rmse,
                "horizon_days": horizon_days,
                "num_items": num_items,
                "store_id": store_id,
                "parameters": json.dumps(parameters) if parameters else None,
            }
        ]
    )

    spark = SparkSession.getActiveSession()
    spark_df = spark.createDataFrame(df)

    (
        spark_df.write.format("jdbc")
        .option("url", get_jdbc_url())
        .option("dbtable", "model_runs")
        .options(**get_jdbc_properties())
        .mode("append")
        .save()
    )
