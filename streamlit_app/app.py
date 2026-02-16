"""
Streamlit Dashboard for M5 Demand Forecasting

Displays historical sales, forecasts, and metrics for selected items.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import smape

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="M5 Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data Loading
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
        "HOBBIES_1_001", "HOBBIES_1_002", "HOBBIES_1_003",
        "HOUSEHOLD_1_001", "HOUSEHOLD_1_002", "HOUSEHOLD_1_003",
        "FOODS_1_001", "FOODS_1_002", "FOODS_1_003", "FOODS_1_004",
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
            records.append({
                "item_id": item,
                "store_id": "CA_1",
                "day_num": day_num,
                "sales": sale,
            })

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
    st.title("📈 M5 Retail Demand Forecasting")
    st.markdown("""
    Interactive dashboard for exploring sales data and forecasts from the
    [Walmart M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) dataset.

    **Store:** CA_1 (California)
    """)

    # Load data
    with st.spinner("Loading data..."):
        df = load_demo_data()

    # Sidebar controls
    st.sidebar.header("Controls")

    items = sorted(df["item_id"].unique())
    selected_item = st.sidebar.selectbox(
        "Select Item",
        items,
        index=0,
    )

    horizon = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=56,
        value=28,
        step=7,
    )

    # Filter data for selected item
    df_item = df[df["item_id"] == selected_item].sort_values("day_num").reset_index(drop=True)

    # Generate forecast
    forecast_df = generate_forecast(df_item, horizon=horizon)

    # Calculate metrics
    smape_value = smape(forecast_df["sales"].values, forecast_df["predicted"].values)
    mae_value = np.abs(forecast_df["sales"] - forecast_df["predicted"]).mean()

    # Layout: metrics row
    st.subheader(f"Item: {selected_item}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SMAPE", f"{smape_value:.1f}%")
    with col2:
        st.metric("MAE", f"{mae_value:.1f}")
    with col3:
        st.metric("Avg Daily Sales", f"{df_item['sales'].mean():.1f}")
    with col4:
        st.metric("Total Days", f"{len(df_item):,}")

    # Historical sales chart
    st.subheader("Historical Sales")

    # Show last N days for readability
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
    st.markdown("""
    **About this demo:**
    - Data: Synthetic data matching M5 schema (demo mode)
    - Model: Simple baseline (moving average + trend)
    - In production: Reads from Azure ADLS, uses trained LightGBM model

    [View on GitHub](https://github.com/ZMTakriti/azure-retail-demand-forecasting)
    """)


if __name__ == "__main__":
    main()
