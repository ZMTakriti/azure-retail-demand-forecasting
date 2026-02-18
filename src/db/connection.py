"""Azure SQL Database connection utilities.

All credentials are read from environment variables — never hardcoded.

Required env vars:
    AZURE_SQL_SERVER   — e.g. your-server.database.windows.net
    AZURE_SQL_DATABASE — e.g. sqldb-m5-forecast
    AZURE_SQL_USER     — e.g. forecast-admin
    AZURE_SQL_PASSWORD — the password
"""

import os


def get_connection():
    """Return a pymssql connection to Azure SQL Database.

    Used by the FastAPI serve layer. pymssql is pure Python — no ODBC driver
    required. Import is lazy so CI tests pass without the package installed.
    """
    import pymssql  # noqa: PLC0415

    server = os.environ["AZURE_SQL_SERVER"]
    database = os.environ["AZURE_SQL_DATABASE"]
    user = os.environ["AZURE_SQL_USER"]
    password = os.environ["AZURE_SQL_PASSWORD"]

    return pymssql.connect(
        server=server,
        user=user,
        password=password,
        database=database,
        port=1433,
        tds_version="7.4",
    )


def get_jdbc_url() -> str:
    """Build a JDBC URL for the Spark JDBC connector."""
    server = os.environ["AZURE_SQL_SERVER"]
    database = os.environ["AZURE_SQL_DATABASE"]
    return (
        f"jdbc:sqlserver://{server}:1433;"
        f"databaseName={database};"
        "encrypt=true;"
        "trustServerCertificate=false;"
    )


def get_jdbc_properties() -> dict:
    """Return JDBC connection properties for the Spark JDBC connector."""
    return {
        "user": os.environ["AZURE_SQL_USER"],
        "password": os.environ["AZURE_SQL_PASSWORD"],
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    }
