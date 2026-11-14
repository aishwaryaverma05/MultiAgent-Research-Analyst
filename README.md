# Multi-Agent Research Analyst

An autonomous, multi-agent AI research analyst application built with **LangGraph**, **LangChain**, **Groq LLM**, **Exa Search**, and **Streamlit**.

The system accepts a research topic and autonomously coordinates specialized agents to plan, search, extract content, synthesize an evidence-grounded research report, rigorously critique it with an independent Critic agent, automatically revise failing drafts, and pause for human-in-the-loop approval before finalization.

---

## Architecture Overview

```
                                      START
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Planner Agent  │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                        ┌─────►│Supervisor Agent │
                        │      └────────┬────────┘
               [Replan] │               │ [Approved]
                        └───────────────┤
                                        ▼
                               ┌─────────────────┐
                               │ Query Generator │
                               └────────┬────────┘
                                        │ (Send dispatch)
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ Parallel Research Workers (Exa API)  │
                     └──────────────────┬───────────────────┘
                                        │ (Reducer merge)
                                        ▼
                               ┌─────────────────┐
                               │ Deduplication & │
                               │ Collection Node │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Reader Agent   │ (Content extraction & cleaning)
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Writer Agent   │ (Structured synthesis & citations)
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                        ┌─────►│  Critic Agent   │ (Independent evaluation 0-10)
                        │      └────────┬────────┘
                        │               │
                        │      [Score-Based Router]
                        │      ├── Score < 7 AND Revisions < 2 ──────┐
                        │      │                                     ▼
                        │      │                            ┌─────────────────┐
                        │      │                            │ Revision Agent  │
                        │      │                            └────────┬────────┘
                        │      │                                     │
                        │      └─────────────────────────────────────┘
                        │
                        ├── Score >= 7 OR Revisions >= 2
                        │
                        ▼
             ┌─────────────────────────┐
             │   Set Quality Warning   │ (Flags if below threshold at revision limit)
             └──────────┬──────────────┘
                        │
                        ▼
             ┌─────────────────────────┐
             │ Human-in-the-Loop Review│ ◄─── (LangGraph interrupt + SQLite checkpoint)
             └──────────┬──────────────┘
                        │
           ┌────────────┼────────────────────────┐
           ▼            ▼                        ▼
      [Approve]     [Reject]         [Request Manual Revision]
           │            │                        │
           ▼            ▼                        ▼
       ┌───────┐    ┌───────┐            ┌────────────────┐
       │  END  │    │  END  │            │ Revision Agent │──► (Back to Critic)
       └───────┘    └───────┘            └────────────────┘
```

---

## Key Features

1. **Autonomous Structured Planning:** Breaks topics into key objectives, prioritized questions, and investigation areas.
2. **Supervisor Gate:** Evaluates research plan clarity and completeness before web queries are dispatched.
3. **Parallel Web Research:** Uses LangGraph's dynamic `Send` mechanism to query Exa in parallel without race conditions.
4. **Resilient Evidence Collection:** Normalizes URLs (stripping tracking parameters like UTM and fbclid), filters private/loopback IPs, and deduplicates sources by relevance score.
5. **Content Extraction:** BeautifulSoup-driven content cleaning with SSRF defenses, size capping, and distinction between full scraped content and search snippets.
6. **Traceable Citations:** Generates reports formatted with inline citations in the format `[Source: Title](URL)` referencing only verified sources.
7. **Strict Independent Critic:** Evaluates reports against supplied evidence across 5 categories (10 points total):
   - Factual Accuracy (0-4)
   - Source Verification (0-2)
   - Evidence Quality (0-2)
   - Completeness & Balance (0-1)
   - Clarity & Structure (0-1)
   - Classifies claims into: `Supported`, `Partially Supported`, `Unsupported`, `Contradicted`.
8. **Automated Quality Revision Loop:** Reports scoring below the threshold (default: 7/10) are automatically routed to the Revision Agent up to the configured limit (default: 2 revisions) before human review.
9. **Human-in-the-Loop Approval:** Pauses execution using LangGraph `interrupt()` and persists state in SQLite (`SqliteSaver`). The reviewer can:
   - **Approve Report:** Finalizes the workflow.
   - **Request Manual Revision:** Passes human feedback to the Revision Agent and routes back through the Critic Agent.
   - **Reject Report:** Terminates workflow with rejection status.
10. **Interactive Streamlit UI:** Complete visibility into research stages, plan, collected sources, full report, score breakdown, and human review actions.

---

## Agent Responsibilities

| Agent | Responsibility | Output Model / Type |
|---|---|---|
| **Planner Agent** | Analyzes topic, decomposes into research questions & focus areas | `ResearchPlan` (Pydantic) |
| **Supervisor Agent** | Validates research plan quality; approves or forces replanning | `SupervisorEvaluation` (Pydantic) |
| **Query Generator** | Generates diverse, search-optimized queries across dimensions | `QuerySet` (Pydantic) |
| **Parallel Workers** | Executes Exa web searches asynchronously across queries | `WorkerResult` (Pydantic) |
| **Collection / Deduplication** | Merges results, normalizes URLs, sorts by relevance score | `list[SourceEvidence]` |
| **Reader Agent** | Fetches HTML, applies anti-SSRF checks, extracts text | `list[SourceEvidence]` (enriched) |
| **Writer Agent** | Synthesizes full 9-section report with inline source citations | Markdown Report string |
| **Critic Agent** | Audits claims against evidence; calculates score out of 10 | `CriticEvaluation` (Pydantic) |
| **Revision Agent** | Re-writes report targeting Critic feedback or user instructions | Revised Report string |

---

## Score-Based Routing Logic

Implemented in `nodes.py:route_after_critic`:

```python
if critic_score >= score_threshold:
    return "human_approval"
if revision_count < max_revisions:
    return "revision"
return "human_approval"  # Reaches human review with quality_warning = True
```

### Scenario Test Matrix (All verified in test suite)

| Scenario | Score | Revision Count | Max Revisions | Route Taken | Quality Warning |
|---|---|---|---|---|---|
| Initial Report Passes | 8 | 0 | 2 | `human_approval` | No |
| Initial Report Fails | 5 | 0 | 2 | `revision` | N/A |
| First Revision Fails | 6 | 1 | 2 | `revision` | N/A |
| Second Revision Passes | 7 | 2 | 2 | `human_approval` | No |
| Second Revision Fails | 5 | 2 | 2 | `human_approval` | **Yes** |
| Exact Threshold | 7 | 0 | 2 | `human_approval` | No |
| Below Threshold at Limit | 6 | 2 | 2 | `human_approval` | **Yes** |

---

## Technology Stack

- **Python 3.11+**
- **LangGraph (1.2+)**: Workflow orchestration, parallel workers (`Send`), interrupts, and checkpointing.
- **LangChain & langchain-groq**: Structured outputs (`with_structured_output`), prompts, and ChatGroq integration.
- **Groq API**: High-speed inference using `llama-3.3-70b-versatile`.
- **Exa Search API**: Targeted neural web search with text extraction and highlights.
- **BeautifulSoup4 & Requests**: Resilient HTML parsing, tag decomposition, and content extraction.
- **Pydantic (v2)**: Strict typed schemas, validation, and auto-correcting score arithmetic.
- **SQLite**: Local persistence for thread checkpoints (`langgraph-checkpoint-sqlite`).
- **Streamlit**: Multi-tab interactive user interface with real-time progress tracking.
- **Pytest**: Comprehensive test suite covering state, tools, critic, routing, and pipeline.

---

## Project Structure

```
MultiAgent-Research-Analyst/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── config.py                 # Centralized configuration & credential checks
├── schemas.py                # Pydantic data models & structured output schemas
├── state.py                  # ResearchState TypedDict & LangGraph reducers
├── tools.py                  # Exa search, URL validation, anti-SSRF, scraping
├── agents.py                 # LLM instances, prompts, and structured output chains
├── nodes.py                  # LangGraph node functions & routing logic
├── pipeline.py               # Graph construction, SQLite checkpointer, run/resume
├── app.py                    # Streamlit frontend application
├── research_checkpoints.sqlite # SQLite database for LangGraph state persistence
└── tests/
    ├── __init__.py           # Package marker
    ├── test_state.py         # State reducers & structure tests
    ├── test_tools.py         # URL safety, anti-SSRF, redirects, deduplication, scraping
    ├── test_critic.py        # Critic schema validation & scoring arithmetic
    ├── test_routing.py       # All mandatory score routing & conditional edges
    ├── test_nodes.py         # Node logic with mocked LLMs
    ├── test_pipeline.py      # Graph compilation & auto-revision flow
    └── test_human_approval.py # Human interrupt, approve/reject/manual revision resume
```

---

## Installation & Setup (Windows)

### 1. Prerequisites
- Python 3.11 or higher installed on your system.
- Git (optional, for version control).

### 2. Clone or Open Workspace
Navigate to the project directory:
```powershell
cd "c:\Users\kprsv\OneDrive\Desktop\Multi-Agent ResearchAnalyst"
```

### 3. Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in your editor and provide your API keys:
```ini
GROQ_API_KEY=your_actual_groq_api_key
EXA_API_KEY=your_actual_exa_api_key

# Optional overrides:
GROQ_MODEL=llama-3.3-70b-versatile
SCORE_THRESHOLD=7
MAX_REVISIONS=2
NUM_SEARCH_QUERIES=5
```

---

## Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Using the App:
1. Enter your research topic in the text area (e.g., *"Current state and future potential of solid-state batteries in electric vehicles"*).
2. Click **Start Research**.
3. Watch the progress bar as the Planner, Supervisor, Query Generator, Parallel Workers, Reader, Writer, and Critic execute.
4. If the Critic score is below 7, watch the Revision Agent automatically refine the draft.
5. Review the report and score breakdown in the **Human Review** interface.
6. Choose:
   - **Approve Report**: Marks the report approved and completes the session.
   - **Request Manual Revision**: Provide specific textual guidance; the Revision Agent revises the report and the Critic evaluates the update.
   - **Reject Report**: Ends the session with rejected status.
7. Download the final report in Markdown format.

---

## Running Tests

Run the full automated test suite (130 tests across 7 test modules):
```powershell
python -m pytest tests/ -v
```

Run a specific test module:
```powershell
# Test Human Approval, interrupt, and resume actions:
python -m pytest tests/test_human_approval.py -v

# Test score-based routing logic:
python -m pytest tests/test_routing.py -v

# Test pipeline construction, checkpointing, and resumption:
python -m pytest tests/test_pipeline.py -v

# Test URL sanitization, anti-SSRF, redirects, and scraping:
python -m pytest tests/test_tools.py -v

# Test Critic scoring, self-verification, and claim models:
python -m pytest tests/test_critic.py -v
```

All 130 unit tests use deterministic mocked responses and **do not consume paid API credits**.

---

## Security & Reliability Design

- **SSRF Protection:** Web scraping rejects private IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.1/8`, `169.254.0.0/16`, `::1`), non-HTTP schemes (`file://`, `ftp://`, `javascript:`), and validates redirect destinations.
- **Resource Constraints:** Web scraping enforces timeout limits (15s), maximum response payloads (500KB), and strip tracking parameters.
- **Reducers on Concurrency:** Shared state updates from parallel research workers use specialized list-merging reducers to prevent race-condition data loss.
- **Resilient Fallbacks:** Individual search worker errors or scraping timeouts log diagnostic messages without crashing the overall research workflow.
- **Deterministic Checkpoints:** Thread IDs ensure sessions survive app reloads and allow pausing for human approval.

---

## Known Limitations & Future Work

- **Dynamic JavaScript Sites:** Sites requiring client-side JS rendering (SPAs) are handled via raw HTTP requests and may yield limited content; adding Playwright/Puppeteer support would expand coverage.
- **PDF Extraction:** Currently optimized for HTML articles; adding PyPDF/PDFPlumber would enable ingesting direct academic PDF URLs.
- **Multi-Model Routing:** Ability to route the Critic and Revision agent to different LLMs (e.g. cross-evaluating Groq drafts with Claude or GPT) for higher evaluation objectivity.

