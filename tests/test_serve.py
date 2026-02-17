"""Tests for the FastAPI serving endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.model.serve import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestForecast:
    @patch("src.model.serve.get_connection")
    def test_forecast_not_found(self, mock_conn):
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        mock_conn.return_value.cursor.return_value = cursor

        resp = client.get("/forecast", params={"item_id": "X", "store_id": "CA_1"})
        assert resp.status_code == 404
        assert "No forecasts found" in resp.json()["detail"]


class TestModelStatus:
    @patch("src.model.serve.get_connection")
    def test_model_status_not_found(self, mock_conn):
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        mock_conn.return_value.cursor.return_value = cursor

        resp = client.get("/model/status")
        assert resp.status_code == 404
        assert "No model runs" in resp.json()["detail"]


class TestForecastItems:
    @patch("src.model.serve.get_connection")
    def test_forecast_items_not_found(self, mock_conn):
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        mock_conn.return_value.cursor.return_value = cursor

        resp = client.get("/forecast/items", params={"store_id": "CA_1"})
        assert resp.status_code == 404
        assert "No items found" in resp.json()["detail"]
