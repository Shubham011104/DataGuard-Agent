# =============================================================================
# config.py — DataGuard Agent: Central Configuration (Groq Only)
# =============================================================================
# All environment variables, threshold constants, and database settings.
# Configured exclusively to use Groq API.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")

# Helper to load secrets from environment variables or Streamlit Cloud Secrets (st.secrets)
def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# ---------------------------------------------------------------------------
# LLM Provider & API Keys (Groq Only)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = "Groq"
GROQ_API_KEY: str = _get_secret("GROQ_API_KEY", "")

LLM_MODEL: str = _get_secret("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(_get_secret("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(_get_secret("LLM_MAX_TOKENS", "2048"))

# ---------------------------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------------------------
# Default: SQLite (zero-config, stored in project root)
DB_PATH: str = os.getenv("DB_PATH", str(Path(__file__).parent / "dataguard.db"))
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ---------------------------------------------------------------------------
# Data Quality Thresholds
# ---------------------------------------------------------------------------
NULL_RATE_THRESHOLD: float = float(os.getenv("NULL_RATE_THRESHOLD", "0.05"))   # 5 %
ZSCORE_THRESHOLD: float = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))           # ±3σ
IQR_MULTIPLIER: float = float(os.getenv("IQR_MULTIPLIER", "1.5"))               # Tukey fence
DUPLICATE_WARN_COUNT: int = int(os.getenv("DUPLICATE_WARN_COUNT", "1"))         # any duplicate

# ---------------------------------------------------------------------------
# Agent Self-Correction Settings
# ---------------------------------------------------------------------------
MAX_CORRECTION_ATTEMPTS: int = int(os.getenv("MAX_CORRECTION_ATTEMPTS", "3"))
AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Sample Dataset Settings
# ---------------------------------------------------------------------------
SAMPLE_TABLE_NAME: str = "ecommerce_transactions"
SAMPLE_ROW_COUNT: int = 500
