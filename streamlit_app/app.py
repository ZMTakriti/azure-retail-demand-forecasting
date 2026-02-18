"""
Streamlit Dashboard for M5 Demand Forecasting

Displays historical sales, forecasts, and metrics for selected items.
Connects to FastAPI when available, falls back to demo data otherwise.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_URL = os.environ.get("FORECAST_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="M5 Demand Forecasting",
    page_icon="\U0001f4c8",
    layout="wide",
)

# -----------------------------------------------------------------------------
# API Helpers
# -----------------------------------------------------------------------------


def _api_available() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        import httpx

        resp = httpx.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=30)
def fetch_items(store_id: str) -> list[str] | None:
    """Fetch available items from the API. Returns None on failure."""
    try:
        import httpx

        resp = httpx.get(f"{API_URL}/forecast/items", params={"store_id": store_id}, timeout=5)
        if resp.status_code == 200:
            return resp.json()["items"]
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def fetch_forecast(item_id: str, store_id: str) -> dict | None:
    """Fetch forecast data from the API. Returns None on failure."""
    try:
        import httpx

        resp = httpx.get(
            f"{API_URL}/forecast",
            params={"item_id": item_id, "store_id": store_id},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def fetch_model_status() -> dict | None:
    """Fetch model status from the API. Returns None on failure."""
    try:
        import httpx

        resp = httpx.get(f"{API_URL}/model/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# Demo Data (fallback)
# -----------------------------------------------------------------------------


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    """
    Generate demo data matching the M5 schema.

    In production, this would read from ADLS or Synapse SQL.
    """
    np.random.seed(42)

    # Sample items from CA_1 store
    items = [
        "HOBBIES_1_001",
        "HOBBIES_1_002",
        "HOBBIES_1_003",
        "HOUSEHOLD_1_001",
        "HOUSEHOLD_1_002",
        "HOUSEHOLD_1_003",
        "FOODS_1_001",
        "FOODS_1_002",
        "FOODS_1_003",
        "FOODS_1_004",
    ]

    days = list(range(1, 1914))  # d_1 to d_1913

    records = []
    for item in items:
        # Generate realistic-ish sales pattern
        base = np.random.randint(5, 50)
        trend = np.linspace(0, np.random.uniform(-5, 10), len(days))
        seasonality = 5 * np.sin(2 * np.pi * np.array(days) / 7)  # Weekly pattern
        noise = np.random.normal(0, 3, len(days))

        sales = np.maximum(0, base + trend + seasonality + noise).astype(int)

        for day_num, sale in zip(days, sales):
            records.append(
                {
                    "item_id": item,
                    "store_id": "CA_1",
                    "day_num": day_num,
                    "sales": sale,
                }
            )

    return pd.DataFrame(records)


@st.cache_data
def generate_forecast(df_item: pd.DataFrame, horizon: int = 28):
    """
    Generate a simple forecast for demonstration.

    Uses a basic approach (moving average + trend) for demo purposes.
    In production, this would load a trained LightGBM model.
    """
    train = df_item.iloc[:-horizon].copy()
    actual = df_item.iloc[-horizon:].copy()

    # Simple forecast: last 28-day average + slight trend
    recent_mean = train["sales"].tail(28).mean()
    recent_trend = (train["sales"].tail(28).mean() - train["sales"].tail(56).head(28).mean()) / 28

    predictions = []
    for i in range(horizon):
        pred = max(0, recent_mean + recent_trend * i + np.random.normal(0, 2))
        predictions.append(pred)

    actual["predicted"] = predictions
    return actual


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------


def main():
    st.title("\U0001f4c8 M5 Retail Demand Forecasting")
    st.markdown(
        """
    Interactive dashboard for exploring sales data and forecasts from the
    [Walmart M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) dataset.

    **Store:** CA_1 (California)
    """
    )

    # Detect API availability
    live_mode = _api_available()

    # Sidebar controls
    st.sidebar.header("Controls")

    if live_mode:
        model_info = fetch_model_status()
        if model_info:
            st.sidebar.markdown("**Model Info**")
            st.sidebar.text(f"Version: {model_info['model_version']}")
            st.sidebar.text(f"Trained: {model_info['trained_at']}")
            st.sidebar.markdown("---")

        api_items = fetch_items("CA_1")
        if api_items:
            items = api_items
        else:
            live_mode = False
            items = sorted(load_demo_data()["item_id"].unique())
    else:
        items = sorted(load_demo_data()["item_id"].unique())

    selected_item = st.sidebar.selectbox(
        "Select Item",
        items,
        index=0,
    )

    if live_mode:
        # ---- Live mode: fetch forecast from API ----
        forecast_data = fetch_forecast(selected_item, "CA_1")

        if forecast_data and forecast_data["forecasts"]:
            forecast_df = pd.DataFrame(forecast_data["forecasts"])
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            st.subheader(f"Item: {selected_item}")
            st.caption(f"Model version: {forecast_data['model_version']}")

            # Forecast chart
            st.subheader(f"Predicted Sales ({len(forecast_df)} days)")
            st.line_chart(
                forecast_df.set_index("date")["predicted_sales"],
                use_container_width=True,
            )

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Avg Predicted Sales",
                    f"{forecast_df['predicted_sales'].mean():.1f}",
                )
            with col2:
                st.metric("Forecast Days", len(forecast_df))
            with col3:
                st.metric("Model Version", forecast_data["model_version"])

            # Detailed table
            with st.expander("View Forecast Details"):
                st.dataframe(forecast_df, use_container_width=True)
        else:
            st.warning("No forecast data available for this item.")
    else:
        # ---- Demo mode: use synthetic data ----
        df = load_demo_data()

        horizon = st.sidebar.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=56,
            value=28,
            step=7,
        )

        df_item = df[df["item_id"] == selected_item].sort_values("day_num").reset_index(drop=True)

        forecast_df = generate_forecast(df_item, horizon=horizon)

        # Calculate metrics
        errors = forecast_df["sales"] - forecast_df["predicted"]
        mae_value = np.abs(errors).mean()
        rmse_value = np.sqrt((errors ** 2).mean())

        st.subheader(f"Item: {selected_item}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{mae_value:.1f}")
        with col2:
            st.metric("RMSE", f"{rmse_value:.1f}")
        with col3:
            st.metric("Avg Daily Sales", f"{df_item['sales'].mean():.1f}")
        with col4:
            st.metric("Total Days", f"{len(df_item):,}")

        # Historical sales chart
        st.subheader("Historical Sales")
        show_days = st.slider("Days to display", 60, 500, 180, step=30)
        df_recent = df_item.tail(show_days)
        st.line_chart(
            df_recent.set_index("day_num")["sales"],
            use_container_width=True,
        )

        # Forecast vs Actual chart
        st.subheader(f"Forecast vs Actual (Last {horizon} Days)")
        forecast_chart_data = forecast_df[["day_num", "sales", "predicted"]].set_index("day_num")
        forecast_chart_data.columns = ["Actual", "Predicted"]
        st.line_chart(forecast_chart_data, use_container_width=True)

        # Detailed forecast table
        with st.expander("View Forecast Details"):
            display_df = forecast_df[["day_num", "sales", "predicted"]].copy()
            display_df["error"] = display_df["predicted"] - display_df["sales"]
            display_df.columns = ["Day", "Actual", "Predicted", "Error"]
            st.dataframe(display_df, use_container_width=True)

    # Footer
    st.markdown("---")
    mode_label = "**Live** (connected to API)" if live_mode else "**Demo** (synthetic data)"
    st.markdown(
        f"""
    **Mode:** {mode_label}

    - Data: {"Live forecasts from Azure SQL" if live_mode else "Synthetic data matching M5 schema (demo mode)"}
    - Model: {"Trained LightGBM via batch prediction pipeline" if live_mode else "Simple baseline (moving average + trend)"}

    [View on GitHub](https://github.com/ZMTakriti/azure-retail-demand-forecasting)
    """
    )


if __name__ == "__main__":
    main()
