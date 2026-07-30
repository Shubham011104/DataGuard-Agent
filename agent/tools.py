# =============================================================================
# agent/tools.py — DataGuard Agent: LangChain Tool Definitions
# =============================================================================
# Defines three custom @tool-decorated functions the ReAct agent can call:
#
#   1. get_database_schema()        — Returns human-readable schema
#   2. execute_sql_query(query)     — Safely executes a read-only SQL query
#   3. get_table_health_summary()   — Returns per-table row/column counts
#
# All tools handle exceptions gracefully and return informative error strings
# so the agent can self-correct without crashing.
# =============================================================================

from __future__ import annotations

import traceback
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

import database as db
import config


# ---------------------------------------------------------------------------
# Shared engine singleton (avoids re-creating the engine on every tool call)
# ---------------------------------------------------------------------------
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = db.get_engine()
    return _engine


def set_engine(engine) -> None:
    """Allow the Streamlit app to inject a specific engine (e.g., user-uploaded DB)."""
    global _engine
    _engine = engine


# ---------------------------------------------------------------------------
# Tool 1: Schema Inspector
# ---------------------------------------------------------------------------

@tool
def get_database_schema() -> str:
    """
    Returns the full database schema: all table names with their column names
    and data types. Always call this first to understand what tables and
    columns are available before writing any SQL query.
    """
    try:
        schema_text = db.get_schema_as_text(engine=_get_engine())
        if not schema_text.strip() or schema_text == "No tables found.":
            return (
                "No tables found in the database. "
                "Please upload a dataset first from the Streamlit UI."
            )
        return f"DATABASE SCHEMA:\n{schema_text}"
    except Exception as exc:
        return f"ERROR retrieving schema: {exc}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Tool 2: Safe SQL Executor
# ---------------------------------------------------------------------------

# Maximum rows returned to avoid flooding the agent's context window
_MAX_RESULT_ROWS = 100

@tool
def execute_sql_query(query: str) -> str:
    """
    Executes a read-only SQL SELECT query against the connected database and
    returns the results as a formatted table string.

    Rules:
    - ONLY SELECT statements are permitted. Any DDL/DML (INSERT, UPDATE,
      DELETE, DROP, CREATE) will be rejected for safety.
    - Results are capped at 100 rows to keep context manageable.
    - If a SQL syntax error occurs, the error message is returned so you can
      correct your query and try again (up to 3 attempts).

    Args:
        query: A valid SQL SELECT statement.

    Returns:
        A string representation of the query result table, or an error message.
    """
    # Safety guard — only allow read queries
    normalized = query.strip().upper()
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
    for kw in forbidden_keywords:
        if normalized.startswith(kw) or f" {kw} " in normalized:
            return (
                f"SAFETY ERROR: '{kw}' statements are not allowed via this tool. "
                "Only SELECT queries are permitted."
            )

    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        return (
            "SAFETY ERROR: Only SELECT or WITH (CTE) statements are allowed. "
            f"Your query starts with: '{query.strip()[:30]}'"
        )

    try:
        df = db.execute_query_to_df(query, engine=_get_engine())

        if df.empty:
            return "Query executed successfully. Result: 0 rows returned (empty result set)."

        # Cap rows for context efficiency
        truncated = len(df) > _MAX_RESULT_ROWS
        display_df = df.head(_MAX_RESULT_ROWS)

        result_str = display_df.to_string(index=False, max_colwidth=60)
        suffix = f"\n\n[Results truncated: showing {_MAX_RESULT_ROWS} of {len(df)} rows]" if truncated else ""

        return (
            f"Query returned {len(df)} row(s), {len(df.columns)} column(s).\n\n"
            f"{result_str}{suffix}"
        )

    except Exception as exc:
        # Return the error so the agent can self-correct
        return (
            f"SQL EXECUTION ERROR:\n"
            f"  Query: {query[:300]}\n"
            f"  Error: {type(exc).__name__}: {exc}\n\n"
            f"Please inspect the error, correct the SQL syntax or column names, "
            f"and call execute_sql_query again with the corrected query."
        )


# ---------------------------------------------------------------------------
# Tool 3: Table Health Summary
# ---------------------------------------------------------------------------

@tool
def get_table_health_summary() -> str:
    """
    Returns a quick health summary for every table in the database:
    row count, column count, and column names. Use this to understand
    dataset sizes before writing analytical queries.
    """
    try:
        engine = _get_engine()
        table_names = db.get_table_names(engine=engine)

        if not table_names:
            return "No tables found in the database."

        lines = ["TABLE HEALTH SUMMARY:", "=" * 40]
        for tbl in table_names:
            try:
                row_count = db.get_row_count(tbl, engine=engine)
                schema = db.get_schema_info(engine=engine)
                cols = schema.get(tbl, [])
                col_names = ", ".join(c["name"] for c in cols)
                lines.append(
                    f"\nTable: {tbl}\n"
                    f"  Rows:    {row_count:,}\n"
                    f"  Columns: {len(cols)}\n"
                    f"  Fields:  {col_names}"
                )
            except Exception as tbl_exc:
                lines.append(f"\nTable: {tbl} — ERROR: {tbl_exc}")

        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR retrieving health summary: {exc}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# All tools list (imported by agent_workflow.py)
# ---------------------------------------------------------------------------
ALL_TOOLS = [get_database_schema, execute_sql_query, get_table_health_summary]
