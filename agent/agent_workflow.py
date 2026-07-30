# =============================================================================
# agent/agent_workflow.py — DataGuard Agent: LangGraph-based Agentic Loop
# =============================================================================
# Uses the modern LangChain 1.x / LangGraph API (create_agent + stream).
# Supports 100% FREE LLM Providers:
#   • Groq API (llama-3.3-70b-versatile, llama-3.1-8b-instant) — FREE
#   • OpenRouter (free models) — FREE
#   • Ollama (local offline LLM) — FREE
#   • OpenAI (gpt-4o-mini, gpt-4o) — Paid
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Ensure parent package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from agent.tools import ALL_TOOLS, set_engine


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are DataGuard, an expert SQL data-quality analyst and autonomous AI agent.

Your goal is to diagnose data quality issues in relational databases by:
1. Inspecting the database schema and table statistics.
2. Writing and executing precise, standard SQL SELECT queries (SQLite-compatible).
3. Interpreting query results to identify root causes of anomalies.
4. Proposing concrete SQL repair scripts (UPDATE / DELETE / CASE WHEN) to fix issues.
5. Providing clear, structured root-cause analysis reports.

## SQL Rules (CRITICAL):
- Use only ANSI/SQLite-compatible SQL syntax.
- For SQLite: use CAST(x AS REAL) not CONVERT(); use strftime() for dates.
- Always alias CTEs clearly (e.g., WITH stats AS (...) SELECT ...).
- Column and table names are case-sensitive — match them exactly as in the schema.
- NEVER write INSERT, UPDATE, DELETE, or DDL — only SELECT for diagnosis.

## Required Output Format:
After investigation, always end with a structured report containing:
- **Root Cause**: What is wrong and why.
- **Affected Records**: How many rows are impacted.
- **SQL Fix Script**: Exact SQL to repair the issue.
- **Prevention Recommendation**: How to prevent this in future.

Think step-by-step. Always call get_database_schema() first.
"""

NL_TO_SQL_PROMPT = """\
You are DataGuard's Natural Language to SQL assistant.

Convert the user's plain-English question about their database into a precise,
executable SQL SELECT query, then run it and show the result.

Rules:
1. Always call get_database_schema() first to see exact table/column names.
2. Write only SELECT queries — never mutate data.
3. Use SQLite-compatible syntax.
4. Return the SQL query clearly formatted in a code block.
5. After writing the query, call execute_sql_query() to run it and show results.
6. If the query fails, diagnose the error, fix the SQL, and retry.
"""


# ---------------------------------------------------------------------------
# Agent Builder
# ---------------------------------------------------------------------------

def _build_llm():
    """Construct the ChatOpenAI model supporting Groq, OpenRouter, Ollama, and OpenAI."""
    from langchain_openai import ChatOpenAI

    provider = getattr(config, "LLM_PROVIDER", "Groq (Free)").lower()

    if "groq" in provider:
        api_key = config.GROQ_API_KEY or config.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "Groq API Key missing. Get a 100% FREE key instantly at: "
                "https://console.groq.com/keys"
            )
        model_name = config.LLM_MODEL if config.LLM_MODEL and "llama" in config.LLM_MODEL or "mixtral" in config.LLM_MODEL else "llama-3.3-70b-versatile"
        return ChatOpenAI(
            model=model_name,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    elif "openrouter" in provider:
        api_key = config.OPENROUTER_API_KEY or config.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OpenRouter API Key missing. Get a FREE key at: "
                "https://openrouter.ai/keys"
            )
        model_name = config.LLM_MODEL if ":" in config.LLM_MODEL else "meta-llama/llama-3.1-8b-instruct:free"
        return ChatOpenAI(
            model=model_name,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    elif "ollama" in provider:
        return ChatOpenAI(
            model=config.LLM_MODEL if config.LLM_MODEL and not config.LLM_MODEL.startswith("gpt") else "llama3.1",
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key="ollama",
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        # Default: OpenAI
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Select 'Groq (Free)' in the sidebar for a 100% free API key!"
            )
        return ChatOpenAI(
            model=config.LLM_MODEL if config.LLM_MODEL.startswith("gpt") else "gpt-4o-mini",
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            api_key=config.OPENAI_API_KEY,
        )


def _build_agent(system_prompt: str = SYSTEM_PROMPT):
    """
    Build a LangChain 1.x agent (backed by LangGraph).
    Returns a CompiledStateGraph that accepts messages-based input.
    """
    from langchain.agents import create_agent
    llm = _build_llm()
    return create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Step Capture
# ---------------------------------------------------------------------------

class AgentStep:
    """Represents one captured step from the agent's reasoning loop."""
    __slots__ = ["thought", "action", "action_input", "observation", "is_final"]

    def __init__(
        self,
        thought: str = "",
        action: str = "",
        action_input: str = "",
        observation: str = "",
        is_final: bool = False,
    ):
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation
        self.is_final = is_final

    def __repr__(self):
        return f"AgentStep(action={self.action!r}, obs={self.observation[:60]!r})"


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_agent_with_steps(
    query: str,
    system_prompt: str = SYSTEM_PROMPT,
    engine=None,
) -> tuple[str, list[AgentStep]]:
    """
    Run the agent on *query* and capture all intermediate steps.

    Parameters
    ----------
    query         : The anomaly description or NL question.
    system_prompt : Override the default system prompt.
    engine        : SQLAlchemy engine to inject into the tool layer.

    Returns
    -------
    (final_answer: str, steps: list[AgentStep])
    """
    if engine is not None:
        set_engine(engine)

    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    agent = _build_agent(system_prompt=system_prompt)

    steps: list[AgentStep] = []
    final_answer = ""

    try:
        # Stream events node-by-node
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="updates",
        ):
            # chunk is a dict: {node_name: state_update}
            for node_name, state_update in chunk.items():
                messages = state_update.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # LLM decided to call a tool
                        for tc in msg.tool_calls:
                            steps.append(AgentStep(
                                thought=getattr(msg, "content", "") or "",
                                action=tc.get("name", ""),
                                action_input=str(tc.get("args", {}).get("query", tc.get("args", ""))),
                                is_final=False,
                            ))
                    elif isinstance(msg, ToolMessage):
                        # Tool returned a result — attach to last step
                        if steps and not steps[-1].observation:
                            steps[-1].observation = str(msg.content)
                        else:
                            steps.append(AgentStep(
                                action=msg.name or "tool_result",
                                observation=str(msg.content),
                                is_final=False,
                            ))
                    elif isinstance(msg, AIMessage) and not msg.tool_calls:
                        # Final text response from the LLM
                        final_answer = str(msg.content)

    except Exception as exc:
        import traceback
        error_detail = traceback.format_exc()
        final_answer = (
            f"**Agent Error**: {type(exc).__name__}: {exc}\n\n"
            f"```\n{error_detail}\n```"
        )

    # Add a final step marker
    steps.append(AgentStep(
        thought=final_answer,
        is_final=True,
    ))

    return final_answer, steps


# ---------------------------------------------------------------------------
# Anomaly Investigation Prompt Builder
# ---------------------------------------------------------------------------

def build_anomaly_prompt(anomaly_record, table_name: str, schema_hint: str = "") -> str:
    """
    Construct a targeted investigation prompt from an AnomalyRecord.
    """
    parts = [
        f"Investigate the following data quality anomaly in table '{table_name}':",
        "",
        f"Anomaly Type: {anomaly_record.anomaly_type}",
        f"Affected Column: {anomaly_record.column}",
        f"Severity: {anomaly_record.severity}",
        f"Description: {anomaly_record.description}",
        f"Estimated Affected Rows: {anomaly_record.affected_rows}",
    ]
    if anomaly_record.suggested_sql:
        parts += [
            "",
            "Initial diagnostic query (verify and expand upon this):",
            anomaly_record.suggested_sql,
        ]
    if schema_hint:
        parts += ["", "Database Schema:", schema_hint]

    parts += [
        "",
        "Please:",
        "1. Call get_database_schema() to confirm schema.",
        "2. Run the diagnostic SQL to confirm and quantify the issue.",
        "3. Investigate the root cause (patterns by date/region/category, etc.).",
        "4. Provide a complete SQL fix script (as a code block).",
        "5. Explain your findings in plain English in the structured format requested.",
    ]
    return "\n".join(parts)
