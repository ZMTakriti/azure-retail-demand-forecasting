"""Azure SQL Database connection utilities.

All credentials are read from environment variables — never hardcoded.

Required env vars:
    AZURE_SQL_SERVER   — e.g. your-server.database.windows.net
    AZURE_SQL_DATABASE — e.g. sqldb-m5-forecast
    AZURE_SQL_USER     — e.g. forecast-admin
    AZURE_SQL_PASSWORD — the password
"""

import os

import pyodbc


def get_connection_string() -> str:
    """Build an ODBC connection string from environment variables."""
    server = os.environ["AZURE_SQL_SERVER"]
    database = os.environ["AZURE_SQL_DATABASE"]
    user = os.environ["AZURE_SQL_USER"]
    password = os.environ["AZURE_SQL_PASSWORD"]

    return (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"Uid={user};"
        f"Pwd={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
    )


def get_connection() -> pyodbc.Connection:
    """Return a pyodbc connection to Azure SQL Database."""
    return pyodbc.connect(get_connection_string())
