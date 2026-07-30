# =============================================================================
# config.py — DataGuard Agent: Central Configuration
# =============================================================================
# All environment variables, threshold constants, and database settings.
# Supports 100% FREE LLM Providers (Groq, OpenRouter, Ollama) as well as OpenAI.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# LLM Provider & API Keys
# ---------------------------------------------------------------------------
# Options: "Groq (Free)", "OpenRouter (Free)", "Ollama (Free Local)", "OpenAI"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "Groq (Free)")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

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
