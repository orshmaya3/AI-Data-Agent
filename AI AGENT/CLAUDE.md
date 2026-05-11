# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Portfolio demo deployed at **ortaeir.com**. Target audience: recruiters.

---

## Quick Start

```bash
pip install -r requirements.txt
python flask_app.py          # runs on http://localhost:5001
```

> `app.py` is a separate Streamlit redirect page — NOT the Flask app. Do not confuse the two.

---

## Project Structure

```
AI AGENT/
├── flask_app.py             # App factory + entry point (port 5001)
├── flask_agents.py          # Singleton manager + per-session agent registry
├── flask_routes/
│   ├── auth.py              # Login, /demo auto-login, session management
│   ├── dashboard.py         # /dashboard + /api/kpis + /api/charts
│   ├── chat.py              # /chat + /api/chat
│   ├── prediction.py        # /prediction + /api/prediction/*
│   ├── consultant.py        # /consultant + /api/consultant/* (agent: Zyon)
│   ├── upload.py            # /api/upload + /api/upload/status + /api/upload/session
│   ├── upload_utils.py      # Column detection, mapping, cleaning (no Flask imports)
│   └── utils.py             # login_required, resolve_manager, resolve_data_agents
├── agents/
│   ├── Manager.py           # Orchestrator — routes all requests
│   ├── Sales_Analyst.py     # Revenue, countries, trends
│   ├── Product_Analyst.py   # Product performance, returns
│   ├── Customer_Analyst.py  # Segmentation, top customers
│   ├── Prediction_Analyst.py# ML forecasting, slow movers, growth
│   ├── Code_Executor.py     # Persistent Python sandbox (subprocess + matplotlib)
│   └── Data_Agent.py        # Startup data loading only
├── flask_templates/         # Jinja2 HTML templates
├── flask_static/            # CSS, JS, avatars
└── data/
    ├── online_retail_II_sampled.parquet  # Primary demo dataset (loaded at startup)
    └── mixed_online_retail.csv           # Legacy format
```

---

## Pages & Routes

| Route | Blueprint | Purpose |
|---|---|---|
| `/` | `flask_app.py` | Landing page |
| `/login` | `auth_bp` | Login form (rate-limited: 5 attempts, 10-min lockout) |
| `/demo` | `auth_bp` | Auto-login as Demo viewer |
| `/dashboard` | `dashboard_bp` | KPI cards + Plotly.js trend charts |
| `/chat` | `chat_bp` | Conversational AI with all agents |
| `/prediction` | `prediction_bp` | ML forecast + product trend charts |
| `/consultant` | `consultant_bp` | Guided 4-phase strategy advisor (Zyon) |

Auth: two hardcoded admin users (`Or`, `Taeir`). Sessions last 30 minutes.  
`before_request` hook in `flask_app.py` assigns a `session['session_id']` UUID to every visitor — required by the upload system.

---

## Agent Architecture

```
User message
  → flask_routes/{chat,consultant}.py
  → utils.resolve_manager(session_id)   ← returns session manager if upload exists, else global
  → Manager._route_to_agent()           ← gpt-4o classifier
  → ReAct agent (LangGraph)             ← gpt-4o
       SalesAnalyst | ProductAnalyst | CustomerAnalyst | PredictionAnalyst | Zyon
  → tools: execute_python + domain-specific pre-built tools
  → Manager yields {"type": "result", "content": ..., "agent_label": ...}
```

- All agents use **gpt-4o**
- `DataAgent` runs at startup only — not invoked per request
- `Manager.handle_request()` → chat agents; `Manager.handle_consultant_request()` → Zyon
- Global demo manager: `from flask_agents import get_manager; manager = get_manager()`
- Session-scoped manager (upload flow): `from flask_agents import get_session_manager; manager = get_session_manager(session_id)`

---

## Data

- Primary source: `data/online_retail_II_sampled.parquet` (loaded at startup)
- Loaded once as a pandas DataFrame, injected into all analyst classes
- Pre-computations (aggregations, ML models) run in each analyst's `__init__` — changing column names breaks everything downstream
- The DataFrame variable name in agent code contexts is always `df`
- Canonical columns: `Customer ID`, `Invoice`, `Description`, `Quantity`, `Price`, `InvoiceDate`, `Country`

---

## Upload / Per-Session Architecture

Users can upload their own CSV/XLSX. Each upload gets an isolated agent pool:

```
POST /api/upload
  → Validate file (50 MB max, 500k rows max, CSV/XLSX/XLS)
  → parse → detect column mapping (upload_utils.detect_column_mapping)
  → If mapping incomplete → return status:"mapping_required" with 5-row preview
  → apply_mapping_and_clean → dedupe, coerce numerics, filter invalid rows
  → register_session_data(session_id, df)
      • Stores df + SalesAnalyst synchronously
      • Spawns background daemon: ManagerAgent.__init__(df)  ← ~10–30 s
  → return { status:"processing" }

Client polls GET /api/upload/status until status == "SESSION_READY"

Chat/Consultant routes call resolve_manager(session_id):
  → If SESSION_READY  → use session's ManagerAgent
  → Otherwise         → fall back to global demo manager
```

Session registry lives in `flask_agents._session_registry` (dict, thread-safe via `_session_lock`).  
Capacity: `MAX_UPLOAD_SESSIONS` env var (default 10).  
Eviction: sessions older than 1 hour, swept every 15 min by a daemon thread started in `create_app()`.

**Column mapping** (`upload_utils.py`):
- `ALL_COLUMNS` / `REQUIRED_COLUMNS` — canonical schema
- `COLUMN_ALIASES` — comprehensive case-insensitive alias dict per canonical column
- `detect_column_mapping(cols)` → `{canonical: matched | None}`
- `mapping_is_complete(mapping)` → bool (Country is optional)
- `apply_mapping_and_clean(df, mapping)` → `(cleaned_df, warnings)`

---

## Consultant — 4-Phase Guided Flow

`/consultant` runs Zyon through a structured UI:

| Phase | What happens |
|---|---|
| 0 — Profile | User enters name, email, business type → stored in `session['business_profile']` via `POST /api/consultant/profile` |
| 1 — Goal | User picks a goal card; `GET /api/consultant/health_preview` returns 3 traffic-light KPIs (MoM growth, churn risk, repeat rate) from pre-loaded analyst data (no LLM call) |
| 1.5 — Questions | 3 goal-specific questions answered by user; answers are appended to the Zyon prompt |
| 2 — Strategy | `POST /api/consultant/analyze` sends full context (profile + goal + answers) to Zyon; follow-up chat via `POST /api/consultant/followup` |

Profile context (`[Context: You are speaking with {name}, who runs a {type} business.]`) is prepended to every `/api/consultant/analyze` call if profile is in session.

---

## Chart Pipeline

Two completely separate chart systems:

### 1. Dashboard & Prediction pages — Plotly.js
- Routes return pre-aggregated JSON via `/api/charts` and `/api/prediction/charts`
- Browser renders with **Plotly.js 2.27.0** — no matplotlib

### 2. Agent-generated charts — matplotlib → base64 PNG

| Step | Location | Detail |
|---|---|---|
| Setup | `Code_Executor.py:18-20` | `matplotlib.use("Agg")` |
| Styling | `Code_Executor.py:104-131` | `_apply_chart_style()` — seaborn whitegrid, DPI 120, (10,5) |
| Execution | `Code_Executor.py:138-218` | `_subprocess_worker()` — child process |
| Capture | `Code_Executor.py:205-218` | `plt.get_fignums()` → `BytesIO` PNG → base64 |
| Buffer | `Code_Executor.py:265, 367-374` | `get_pending_charts()` clears and returns list |

**Known gap:** `flask_routes/chat.py:api_chat()` never calls `get_pending_charts()` — agent-generated charts during chat are silently discarded.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | All agent LLM calls |
| `SECRET_KEY` / `FLASK_SECRET_KEY` | No | Flask session signing (fallback exists) |
| `MAX_UPLOAD_SESSIONS` | No | Upload session capacity (default 10) |

---

## Tech Stack

**Backend:** Flask, LangChain + LangGraph, OpenAI SDK (gpt-4o), pandas, numpy, matplotlib (Agg), seaborn, python-dotenv

**Frontend:** Plotly.js 2.27.0 (in `base.html`), marked.js (markdown in chat), vanilla JS, CSS custom properties for dark/light theme. All CSS is **inline `<style>` tags** in each template — no external stylesheet files.

**No tests. No linting tools.**

---

## Response Style Preference

Keep responses **short and direct**. No summaries at the end of turns.
