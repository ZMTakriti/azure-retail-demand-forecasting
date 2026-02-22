"""FastAPI application for serving demand forecasts.

Reads pre-computed predictions from the Azure SQL DB prediction store.
Deploy to Azure App Service (F1 free tier).

Run locally:
    uvicorn src.model.serve:app --reload
"""

from fastapi import FastAPI, HTTPException, Query

from src.db.connection import get_connection

app = FastAPI(
    title="M5 Demand Forecast API",
    description="REST API serving batch forecasts from the prediction store.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/forecast")
def get_forecast(
    item_id: str = Query(..., description="Item identifier, e.g. FOODS_3_090"),
    store_id: str = Query(default="CA_1", description="Store identifier"),
) -> dict:
    """Return the latest forecasts for a given item and store."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT forecast_date, predicted_sales, model_version "
        "FROM ("
        "  SELECT forecast_date, predicted_sales, model_version,"
        "    ROW_NUMBER() OVER (PARTITION BY forecast_date ORDER BY id DESC) AS rn"
        "  FROM forecasts"
        "  WHERE item_id = %s AND store_id = %s"
        "    AND model_version = (SELECT TOP 1 model_version FROM model_runs WHERE is_active = 1 ORDER BY trained_at DESC)"
        ") t WHERE rn = 1 "
        "ORDER BY forecast_date",
        (item_id, store_id),
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No forecasts found for this item/store.")

    return {
        "item_id": item_id,
        "store_id": store_id,
        "model_version": rows[0][2],
        "forecasts": [{"date": str(row[0]), "predicted_sales": row[1]} for row in rows],
    }


@app.get("/forecast/items")
def get_forecast_items(
    store_id: str = Query(default="CA_1", description="Store identifier"),
    sort_by: str = Query(
        default="name", description="Sort order: 'name' (alphabetical) or 'volume' (avg sales desc)"
    ),
) -> dict:
    """Return items with forecasts for a given store.

    sort_by='name' returns alphabetical order (default).
    sort_by='volume' joins sales_history and orders by avg actual_sales descending.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if sort_by == "volume":
        cursor.execute(
            "SELECT f.item_id "
            "FROM ("
            "  SELECT DISTINCT item_id FROM forecasts"
            "  WHERE store_id = %s"
            "    AND model_version = (SELECT TOP 1 model_version FROM model_runs WHERE is_active = 1 ORDER BY trained_at DESC)"
            ") f "
            "LEFT JOIN ("
            "  SELECT item_id, AVG(actual_sales) AS avg_sales"
            "  FROM sales_history"
            "  WHERE store_id = %s"
            "    AND model_version = (SELECT TOP 1 model_version FROM model_runs WHERE is_active = 1 ORDER BY trained_at DESC)"
            "  GROUP BY item_id"
            ") h ON f.item_id = h.item_id "
            "ORDER BY COALESCE(h.avg_sales, 0) DESC",
            (store_id, store_id),
        )
    else:
        cursor.execute(
            "SELECT DISTINCT item_id FROM forecasts "
            "WHERE store_id = %s "
            "  AND model_version = (SELECT TOP 1 model_version FROM model_runs WHERE is_active = 1 ORDER BY trained_at DESC) "
            "ORDER BY item_id",
            (store_id,),
        )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No items found for this store.")

    return {
        "store_id": store_id,
        "items": [row[0] for row in rows],
    }


@app.get("/forecast/history")
def get_forecast_history(
    item_id: str = Query(..., description="Item identifier, e.g. FOODS_3_090"),
    store_id: str = Query(default="CA_1", description="Store identifier"),
) -> dict:
    """Return validation-period actual sales for a given item and store."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT sale_date, actual_sales "
        "FROM ("
        "  SELECT sale_date, actual_sales,"
        "    ROW_NUMBER() OVER (PARTITION BY sale_date ORDER BY id DESC) AS rn"
        "  FROM sales_history"
        "  WHERE item_id = %s AND store_id = %s"
        "    AND model_version = (SELECT TOP 1 model_version FROM model_runs WHERE is_active = 1 ORDER BY trained_at DESC)"
        ") t WHERE rn = 1 "
        "ORDER BY sale_date",
        (item_id, store_id),
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No history found for this item/store.")

    return {
        "item_id": item_id,
        "store_id": store_id,
        "history": [{"date": str(row[0]), "actual_sales": row[1]} for row in rows],
    }


@app.get("/model/status")
def model_status() -> dict:
    """Return metadata for the latest active model run."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT TOP 1 model_version, trained_at, mae, rmse, weighted_mae, horizon_days, num_items, store_id "
        "FROM model_runs WHERE is_active = 1 ORDER BY trained_at DESC"
    )

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="No model runs found.")

    return {
        "model_version": row[0],
        "trained_at": str(row[1]),
        "mae": row[2],
        "rmse": row[3],
        "weighted_mae": row[4],
        "horizon_days": row[5],
        "num_items": row[6],
        "store_id": row[7],
    }
