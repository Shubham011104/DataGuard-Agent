# =============================================================================
# database.py — DataGuard Agent: Database Layer
# =============================================================================
# Handles:
#   • SQLAlchemy engine creation (SQLite / PostgreSQL)
#   • Sample e-commerce dataset generation with intentional data quality issues
#   • CSV → SQLite ingestion
#   • Schema introspection helpers
# =============================================================================

import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column, Float, Integer, MetaData, String, Table, Text,
    create_engine, inspect, text,
)
from sqlalchemy.orm import sessionmaker

import config


# ---------------------------------------------------------------------------
# Engine / Session Factory
# ---------------------------------------------------------------------------

def get_engine(database_url: str = config.DATABASE_URL):
    """Create and return a SQLAlchemy engine."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, echo=False)


def get_session(engine=None):
    """Return a new SQLAlchemy session bound to *engine*."""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


# ---------------------------------------------------------------------------
# Sample Dataset Generator
# ---------------------------------------------------------------------------

def generate_sample_dataset(n_rows: int = config.SAMPLE_ROW_COUNT, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic e-commerce transactions DataFrame with intentional
    data quality issues for demonstration:

    Issues injected:
      • ~8 % null values in `customer_email` (exceeds 5 % threshold)
      • ~3 % null values in `shipping_cost`
      • Duplicate transaction IDs (primary-key violation)
      • Revenue outliers (Z-score > 3)
      • Negative quantity values (range violation)
      • Inconsistent category labels (mixed case / typos)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # -- Base columns ---------------------------------------------------------
    transaction_ids = [f"TXN{i:05d}" for i in range(1, n_rows + 1)]

    # Inject duplicate PKs (≈2 % rows)
    dup_indices = rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)
    for idx in dup_indices:
        transaction_ids[idx] = transaction_ids[max(0, idx - 1)]

    categories_clean = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]
    # Introduce typos / mixed-case anomalies
    categories_dirty = categories_clean + [
        "electronics", "CLOTHING", "Bookss", "home & garden", "Sprots"
    ]
    categories = rng.choice(categories_dirty, size=n_rows).tolist()

    # Base revenue: normally distributed around $120
    revenue = rng.normal(loc=120, scale=40, size=n_rows)

    # Inject high outliers (Z-score >> 3)
    outlier_mask = rng.random(n_rows) < 0.02
    revenue[outlier_mask] = rng.uniform(500, 2000, size=outlier_mask.sum())

    # Inject negative revenue outliers
    neg_mask = rng.random(n_rows) < 0.01
    revenue[neg_mask] = rng.uniform(-300, -10, size=neg_mask.sum())

    revenue = np.round(revenue, 2)

    quantity = rng.integers(1, 20, size=n_rows).tolist()
    # Inject negative quantity values
    neg_qty = rng.choice(n_rows, size=max(1, n_rows // 100), replace=False)
    for i in neg_qty:
        quantity[i] = -rng.integers(1, 5)

    shipping_cost = np.round(rng.uniform(2.5, 25.0, size=n_rows), 2)
    # Inject nulls into shipping_cost (~3 %)
    null_ship = rng.random(n_rows) < 0.03
    shipping_cost = shipping_cost.astype(object)
    shipping_cost[null_ship] = None

    # Customer emails with ~8 % nulls
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org"]
    emails = [
        f"user{''.join(random.choices(string.ascii_lowercase, k=5))}@{random.choice(domains)}"
        for _ in range(n_rows)
    ]
    null_email = rng.random(n_rows) < 0.08
    emails = [None if null_email[i] else emails[i] for i in range(n_rows)]

    # Order dates over the last 2 years
    base_date = datetime(2023, 1, 1)
    order_dates = [
        (base_date + timedelta(days=int(rng.integers(0, 730)))).strftime("%Y-%m-%d")
        for _ in range(n_rows)
    ]

    regions = rng.choice(["North", "South", "East", "West", "Central"], size=n_rows).tolist()
    payment_methods = rng.choice(
        ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash"], size=n_rows
    ).tolist()

    df = pd.DataFrame({
        "transaction_id":  transaction_ids,
        "order_date":      order_dates,
        "category":        categories,
        "revenue":         revenue,
        "quantity":        quantity,
        "shipping_cost":   shipping_cost,
        "customer_email":  emails,
        "region":          regions,
        "payment_method":  payment_methods,
    })

    return df


# ---------------------------------------------------------------------------
# Load CSV / DataFrame into Database
# ---------------------------------------------------------------------------

def load_dataframe_to_db(
    df: pd.DataFrame,
    table_name: str,
    engine=None,
    if_exists: str = "replace",
) -> None:
    """Write *df* to a SQL table using pandas `to_sql`."""
    if engine is None:
        engine = get_engine()
    df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
    print(f"[database] Loaded {len(df):,} rows -> table '{table_name}'")


def load_csv_to_db(csv_path: str, table_name: str, engine=None) -> pd.DataFrame:
    """Read a CSV file and write it to the database.  Returns the DataFrame."""
    df = pd.read_csv(csv_path)
    load_dataframe_to_db(df, table_name, engine=engine)
    return df


def seed_sample_database(engine=None) -> pd.DataFrame:
    """Generate and persist the sample dataset. Returns the DataFrame."""
    if engine is None:
        engine = get_engine()
    df = generate_sample_dataset()
    load_dataframe_to_db(df, config.SAMPLE_TABLE_NAME, engine=engine)
    return df


# ---------------------------------------------------------------------------
# Schema Introspection
# ---------------------------------------------------------------------------

def get_table_names(engine=None) -> list[str]:
    """Return all table names in the connected database."""
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    return inspector.get_table_names()


def get_schema_info(engine=None) -> dict[str, list[dict]]:
    """
    Return a dict mapping each table name → list of column dicts:
      { "name": str, "type": str, "nullable": bool }
    """
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    schema: dict[str, list[dict]] = {}
    for table_name in inspector.get_table_names():
        cols = []
        for col in inspector.get_columns(table_name):
            cols.append({
                "name":     col["name"],
                "type":     str(col["type"]),
                "nullable": col.get("nullable", True),
            })
        schema[table_name] = cols
    return schema


def get_schema_as_text(engine=None) -> str:
    """Return the database schema as a human-readable string (for LLM prompts)."""
    schema = get_schema_info(engine=engine)
    lines = []
    for table, cols in schema.items():
        lines.append(f"Table: {table}")
        for col in cols:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            lines.append(f"  - {col['name']} ({col['type']}, {nullable})")
    return "\n".join(lines) if lines else "No tables found."


def execute_query_to_df(query: str, engine=None) -> pd.DataFrame:
    """Execute a raw SQL SELECT query and return results as a DataFrame."""
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn)


def get_row_count(table_name: str, engine=None) -> int:
    """Return the row count for a given table."""
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()
