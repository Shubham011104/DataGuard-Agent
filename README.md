# 🛡️ DataGuard Agent
### Autonomous Data Quality & Anomaly Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20%2F%20LangGraph-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

An intelligent, end-to-end agentic data auditing pipeline that connects to relational databases, automatically runs statistical anomaly detection ($Z$-score, Tukey IQR fences, null rates, duplicate primary keys, category drift), and deploys an autonomous **ReAct AI Agent** (Reason + Act) to dynamically write diagnostic SQL, execute queries, self-correct syntax errors, and generate root-cause analysis reports with exact SQL fix scripts.

---

## ✨ Key Features

* **📊 Automated Statistical Profiling:** Baseline statistics (mean, std dev, min/max, skewness, quartiles) and 5 anomaly detectors ($Z$-Score $\pm 3\sigma$, Tukey IQR fences, null rates > 5%, primary key duplicates, mixed-case typos).
* **🤖 Autonomous ReAct Agent Loop:** Powered by **LangGraph**, the agent reasons step-by-step (*Thought $\rightarrow$ SQL Execution $\rightarrow$ Observation $\rightarrow$ Self-Correction*) to investigate flagged anomalies.
* **🔄 SQL Self-Correction:** Automatically catches SQL execution tracebacks, inspects table schemas, and retries up to 3 times to correct syntax or column name errors.
* **💬 Natural Language → SQL Copilot:** An interactive chat interface for non-technical users to query the database using plain English.
* **🌟 100% FREE LLM Support:** Works out of the box with **Groq Cloud API** (free `llama-3.3-70b-versatile`), **OpenRouter**, **Ollama** (local offline), and **OpenAI**.
* **🔒 Read-Only Execution Safety Guard:** Built-in regex guards block DML/DDL operations (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`) during agent diagnosis.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Database Layer** | SQLite / PostgreSQL via `SQLAlchemy` ORM |
| **Analytics & Data Processing** | `pandas`, `numpy`, `scipy` |
| **Agentic Framework** | `langchain`, `langgraph`, `pydantic` |
| **LLM Providers** | Groq (`llama-3.3-70b-versatile`), OpenRouter, Ollama, OpenAI |
| **Frontend UI & Visualization** | `streamlit`, `plotly` |

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/dataguard-agent.git
cd dataguard-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Copy `.env.example` to `.env`:
```bash
# On Linux/macOS
cp .env.example .env

# On Windows (Command Prompt)
copy .env.example .env
```

### 4. Configure Your LLM Credentials (100% FREE)

DataGuard supports **100% FREE** cloud & local AI providers:

* **Groq Cloud (Recommended, 100% FREE):**
  Get a free API key at **[console.groq.com/keys](https://console.groq.com/keys)** and set:
  ```env
  GROQ_API_KEY=gsk_your_free_groq_key_here
  ```
---

## 🚀 Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.

---

## 🖥️ Dashboard Overview & Tabs

1. **📊 Tab 1: Data Overview & Health Check**
   * Load the synthetic 500-row e-commerce dataset (or upload your own CSV).
   * Run full statistical profiling to generate column metrics, Plotly distribution charts, null rate heatmaps, and flagged anomaly cards.

2. **🤖 Tab 2: Autonomous Agent Audit**
   * Select a flagged anomaly from Tab 1.
   * Click **Launch Agent Investigation** to observe real-time agent reasoning steps (Thought, Action, Observation, Self-Correction) and view the final SQL `UPDATE`/`DELETE` fix script.

3. **💬 Tab 3: Natural Language → SQL Copilot**
   * Ask plain-English questions about your database (e.g., *"What are the top 5 categories by total revenue?"*).
   * The agent converts your question to SQL, executes it against SQLite, and displays formatted results.

---

## 📁 Repository Structure

```text
dataguard_agent/
├── config.py              # Configuration, thresholds, and LLM provider settings
├── database.py            # SQLAlchemy engine, schema introspection, sample dataset generator
├── profiling.py           # Statistical profiling engine (Z-Score, IQR, Nulls, Duplicates, Typos)
├── agent/
│   ├── __init__.py        # Package marker
│   ├── tools.py           # Custom LangChain @tool wrappers (Schema, Safe SQL Executor, Health)
│   └── agent_workflow.py  # LangGraph ReAct agent loop with self-correction
├── app.py                 # Streamlit UI dashboard
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── .gitignore             # Excluded files for Git
└── README.md              # Project documentation
```

---

## 🔒 Security & Safety Notes

* The `execute_sql_query` tool strictly enforces **read-only SELECT queries**.
* Any attempt by an LLM to generate DML/DDL statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`) is rejected by regular expression safeguards.
* Generated SQL repair scripts are presented to the user for human review, not executed automatically.

---
