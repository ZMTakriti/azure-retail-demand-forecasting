"""
Streamlit Dashboard for M5 Demand Forecasting

Displays historical sales, forecasts, and department analytics.
Connects to FastAPI when available, falls back to demo data otherwise.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_URL = os.environ.get("FORECAST_API_URL", "http://localhost:8000")

DEPT_COLORS = {"FOODS": "#2ca02c", "HOUSEHOLD": "#1f77b4", "HOBBIES": "#ff7f0e"}

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
def fetch_items(store_id: str, sort_by: str = "name", dept: str | None = None) -> list[str] | None:
    """Fetch available items from the API. Returns None on failure."""
    try:
        import httpx

        params: dict = {"store_id": store_id, "sort_by": sort_by}
        if dept:
            params["dept"] = dept
        resp = httpx.get(f"{API_URL}/forecast/items", params=params, timeout=5)
        if resp.status_code == 200:
            return resp.json()["items"]
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def fetch_departments(store_id: str) -> list[dict] | None:
    """Fetch department summaries from the API. Returns None on failure."""
    try:
        import httpx

        resp = httpx.get(
            f"{API_URL}/forecast/departments", params={"store_id": store_id}, timeout=5
        )
        if resp.status_code == 200:
            return resp.json()["departments"]
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
def fetch_history(item_id: str, store_id: str) -> dict | None:
    """Fetch validation-period actuals from the API. Returns None on failure."""
    try:
        import httpx

        resp = httpx.get(
            f"{API_URL}/forecast/history",
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
    """Generate synthetic demo data matching the M5 schema."""
    np.random.seed(42)

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

    # Realistic volume differences by department (FOODS >> HOUSEHOLD >> HOBBIES)
    base_by_dept = {"HOBBIES": (2, 15), "HOUSEHOLD": (5, 25), "FOODS": (15, 60)}
    days = list(range(1, 1914))

    records = []
    for item in items:
        dept = item.split("_")[0]
        lo, hi = base_by_dept[dept]
        base = np.random.randint(lo, hi)
        trend = np.linspace(0, np.random.uniform(-3, 5), len(days))
        seasonality = (base * 0.3) * np.sin(2 * np.pi * np.array(days) / 7)
        noise = np.random.normal(0, base * 0.15, len(days))
        sales = np.maximum(0, base + trend + seasonality + noise).astype(int)
        for day_num, sale in zip(days, sales):
            records.append({"item_id": item, "store_id": "CA_1", "day_num": day_num, "sales": sale})

    return pd.DataFrame(records)


@st.cache_data
def generate_forecast(df_item: pd.DataFrame, horizon: int = 28) -> pd.DataFrame:
    """Generate a simple moving-average forecast for demo mode."""
    train = df_item.iloc[:-horizon].copy()
    actual = df_item.iloc[-horizon:].copy()
    recent_mean = train["sales"].tail(28).mean()
    recent_trend = (
        train["sales"].tail(28).mean() - train["sales"].tail(56).head(28).mean()
    ) / 28
    predictions = [
        max(0, recent_mean + recent_trend * i + np.random.normal(0, recent_mean * 0.1))
        for i in range(horizon)
    ]
    actual = actual.copy()
    actual["predicted"] = predictions
    return actual


# -----------------------------------------------------------------------------
# Chart helpers
# -----------------------------------------------------------------------------


def _dept_bar_chart(dept_data: list[dict]) -> go.Figure:
    """Horizontal bar chart of avg daily forecast by department."""
    depts = [d["dept"] for d in dept_data]
    vals = [d.get("avg_daily_forecast", d.get("avg_sales", 0)) for d in dept_data]
    counts = [d["item_count"] for d in dept_data]
    colors = [DEPT_COLORS.get(d, "#9467bd") for d in depts]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=depts,
            orientation="h",
            text=[f"{v:.1f} units/day  ({c} items)" for v, c in zip(vals, counts)],
            textposition="auto",
            marker_color=colors,
        )
    )
    fig.update_layout(
        xaxis_title="Avg Daily Forecast (units)",
        height=180,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


def _actual_vs_predicted_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    x_col: str = "date",
) -> go.Figure:
    """Plotly actual vs predicted line chart with clean date axis."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df[x_col],
            y=history_df["actual_sales"] if "actual_sales" in history_df.columns else history_df["sales"],
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
        )
    )
    pred_col = "predicted_sales" if "predicted_sales" in forecast_df.columns else "predicted"
    fig.add_trace(
        go.Scatter(
            x=forecast_df[x_col],
            y=forecast_df[pred_col],
            mode="lines",
            name="Predicted",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    x_axis = dict(tickformat="%b %d", showgrid=True) if x_col == "date" else dict(title="Day")
    fig.update_layout(
        xaxis=x_axis,
        yaxis=dict(title="Units / day"),
        hovermode="x unified",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _history_chart(df: pd.DataFrame) -> go.Figure:
    """Simple historical sales line chart for demo mode."""
    fig = go.Figure(
        go.Scatter(
            x=df["day_num"],
            y=df["sales"],
            mode="lines",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Units / day",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------


def main():
    st.title("\U0001f4c8 M5 Retail Demand Forecasting")
    st.markdown(
        "Interactive dashboard for demand forecasts from the "
        "[Walmart M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) dataset. "
        "**Store:** CA_1 (California)"
    )

    live_mode = _api_available()
    st.sidebar.header("Controls")

    # ------------------------------------------------------------------
    # Sidebar — Live mode
    # ------------------------------------------------------------------
    if live_mode:
        model_info = fetch_model_status()
        if model_info:
            st.sidebar.markdown("**Model Info**")
            st.sidebar.text(f"Version: {model_info['model_version']}")
            st.sidebar.text(f"Trained: {model_info['trained_at'][:10]}")
            if model_info.get("mae") is not None:
                st.sidebar.text(f"MAE:     {model_info['mae']:.2f} units/day")
            if model_info.get("rmse") is not None:
                st.sidebar.text(f"RMSE:    {model_info['rmse']:.2f} units/day")
            if model_info.get("weighted_mae") is not None:
                st.sidebar.text(f"W-MAE:   {model_info['weighted_mae']:.2f} units/day")
            st.sidebar.markdown("---")

        dept_data = fetch_departments("CA_1")
        dept_names = [d["dept"] for d in dept_data] if dept_data else []
        selected_dept = st.sidebar.selectbox("Department", ["All"] + dept_names)
        dept_filter = None if selected_dept == "All" else selected_dept

        sort_by_volume = st.sidebar.checkbox("Sort items by sales volume", value=False)
        sort_by = "volume" if sort_by_volume else "name"

        api_items = fetch_items("CA_1", sort_by=sort_by, dept=dept_filter)
        if api_items:
            items = api_items
        else:
            live_mode = False
            items = sorted(load_demo_data()["item_id"].unique())

    # ------------------------------------------------------------------
    # Sidebar — Demo mode
    # ------------------------------------------------------------------
    else:
        dept_data = None
        selected_dept = st.sidebar.selectbox("Department", ["All", "FOODS", "HOUSEHOLD", "HOBBIES"])
        horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 56, 28, step=7)
        df_demo = load_demo_data()
        all_items = sorted(df_demo["item_id"].unique())
        items = (
            [i for i in all_items if i.startswith(selected_dept)]
            if selected_dept != "All"
            else all_items
        )

    selected_item = st.sidebar.selectbox("Select Item", items)

    # ------------------------------------------------------------------
    # Main content — Live mode
    # ------------------------------------------------------------------
    if live_mode:
        # Department overview bar chart
        if dept_data:
            st.subheader("Department Overview")
            st.plotly_chart(_dept_bar_chart(dept_data), use_container_width=True)
            st.markdown("---")

        forecast_data = fetch_forecast(selected_item, "CA_1")

        if forecast_data and forecast_data["forecasts"]:
            forecast_df = pd.DataFrame(forecast_data["forecasts"])
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            st.subheader(f"Item: {selected_item}")
            st.caption(f"Model version: {forecast_data['model_version']}")

            history_data = fetch_history(selected_item, "CA_1")

            if history_data and history_data["history"]:
                history_df = pd.DataFrame(history_data["history"])
                history_df["date"] = pd.to_datetime(history_df["date"])

                merged = history_df.merge(forecast_df, on="date")
                errors = merged["actual_sales"] - merged["predicted_sales"]
                mae_val = float(np.abs(errors).mean())
                rmse_val = float(np.sqrt((errors**2).mean()))

                st.subheader("Actual vs Predicted — Holdout Period")
                st.plotly_chart(
                    _actual_vs_predicted_chart(history_df, forecast_df),
                    use_container_width=True,
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{mae_val:.2f} units/day")
                with col2:
                    st.metric("RMSE", f"{rmse_val:.2f} units/day")
                with col3:
                    st.metric("Avg Actual Sales", f"{history_df['actual_sales'].mean():.1f}")

            else:
                st.subheader(f"Predicted Sales — {len(forecast_df)} days")
                fig = go.Figure(
                    go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["predicted_sales"],
                        mode="lines",
                        line=dict(color="#ff7f0e", width=2),
                    )
                )
                fig.update_layout(
                    xaxis=dict(tickformat="%b %d"),
                    yaxis_title="Units / day",
                    height=360,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Predicted Sales", f"{forecast_df['predicted_sales'].mean():.1f}")
                with col2:
                    st.metric("Forecast Days", len(forecast_df))

            with st.expander("View Forecast Details"):
                st.dataframe(forecast_df, use_container_width=True)

        else:
            st.warning("No forecast data available for this item.")

    # ------------------------------------------------------------------
    # Main content — Demo mode
    # ------------------------------------------------------------------
    else:
        df_demo = load_demo_data()

        # Department overview from synthetic data
        dept_summary = (
            df_demo.assign(dept=df_demo["item_id"].str.split("_").str[0])
            .groupby("dept")
            .agg(avg_daily_forecast=("sales", "mean"), item_count=("item_id", "nunique"))
            .reset_index()
            .sort_values("avg_daily_forecast", ascending=False)
            .to_dict("records")
        )
        st.subheader("Department Overview (Demo)")
        st.plotly_chart(_dept_bar_chart(dept_summary), use_container_width=True)
        st.markdown("---")

        df_item = (
            df_demo[df_demo["item_id"] == selected_item]
            .sort_values("day_num")
            .reset_index(drop=True)
        )
        forecast_df = generate_forecast(df_item, horizon=horizon)

        errors = forecast_df["sales"] - forecast_df["predicted"]
        mae_value = float(np.abs(errors).mean())
        rmse_value = float(np.sqrt((errors**2).mean()))

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

        st.subheader("Historical Sales")
        show_days = st.slider("Days to display", 60, 500, 180, step=30)
        st.plotly_chart(_history_chart(df_item.tail(show_days)), use_container_width=True)

        st.subheader(f"Forecast vs Actual (Last {horizon} Days)")
        st.plotly_chart(
            _actual_vs_predicted_chart(
                forecast_df.rename(columns={"sales": "actual_sales"}),
                forecast_df,
                x_col="day_num",
            ),
            use_container_width=True,
        )

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

    - Data: {"Live forecasts from Azure SQL" if live_mode else "Synthetic data matching M5 schema"}
    - Model: {"Trained LightGBM via batch prediction pipeline" if live_mode else "Simple baseline (moving average + trend)"}

    [View on GitHub](https://github.com/ZMTakriti/azure-retail-demand-forecasting)
    """
    )


if __name__ == "__main__":
    main()
