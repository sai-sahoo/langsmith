# LangSmith Tracing: Hands-on Examples

A set of small Python scripts that build progressively more complex LLM applications with **LangChain** and **LangGraph**, and trace every run with **[LangSmith](https://smith.langchain.com/)**.

The scripts are numbered so you can read them in order. Each one adds a new concept, going from a single LLM call to a RAG pipeline, a tool-using agent, and a multi-node LangGraph workflow. Each also shows a different way to shape what shows up in the LangSmith UI: project names, run names, tags, metadata, custom spans, and nested traces.

---

## Table of Contents

- [What You'll Learn](#what-youll-learn)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [The Examples](#the-examples)
  - [1. Simple LLM Call](#1-simple-llm-call--1_simple_llm_callpy)
  - [2. Sequential Chain](#2-sequential-chain--2_sequential_chainpy)
  - [3. PDF RAG Chatbot (v1 → v4)](#3-pdf-rag-chatbot-v1--v4)
  - [4. ReAct Agent with Tools](#4-react-agent-with-tools--4_agentpy)
  - [5. LangGraph Essay Evaluator](#5-langgraph-essay-evaluator--5_langgraphpy)
- [LangSmith Tracing Techniques Used](#langsmith-tracing-techniques-used)
- [LangSmith Projects Created](#langsmith-projects-created)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)

---

## What You'll Learn

- Composing LangChain pipelines with the LCEL pipe syntax (`prompt | model | parser`)
- Automatic LangSmith tracing of LangChain and LangGraph runs through environment variables
- Routing runs to separate LangSmith **projects** with `LANGCHAIN_PROJECT`
- Adding `run_name`, `tags` and `metadata` to runs via `config`
- Tracing plain Python functions with the `@traceable` decorator
- Nesting spans so that setup and query steps appear under one root trace
- Caching a FAISS vector index on disk, with explicit traces for cache hits (`load_index`) and misses (`build_index`)
- Building a ReAct agent with web search and a custom tool
- Building a parallel fan-out/fan-in LangGraph workflow with structured (Pydantic) LLM output

---

## Project Structure

```
.
├── 1_simple_llm_call.py     # Minimal prompt → model → parser chain
├── 2_sequential_chain.py    # Two-model chain: generate report → summarize
├── 3_rag_v1.py              # PDF RAG, auto-traced LangChain only
├── 3_rag_v2.py              # + @traceable spans for setup steps
├── 3_rag_v3.py              # + single root trace wrapping setup + query
├── 3_rag_v4.py              # + on-disk FAISS index cache keyed by content hash
├── 4_agent.py               # ReAct agent with DuckDuckGo search + weather tool
├── 5_langgraph.py           # LangGraph parallel essay evaluator
├── main.py                  # Placeholder entry point (uv project template)
├── islr.pdf                 # Sample document for the RAG examples
├── .indices/                # Cached FAISS indexes created by 3_rag_v4.py
├── pyproject.toml           # Project metadata and pinned dependencies
├── uv.lock                  # uv lockfile
└── .python-version          # Python 3.12
```

> `islr.pdf` is *An Introduction to Statistical Learning*, used as the knowledge base for the RAG examples. You can replace it with any PDF by changing `PDF_PATH` in the `3_rag_*.py` scripts.

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`
- An **OpenAI API key**, used for chat models and embeddings
- A **LangSmith API key**, available free at [smith.langchain.com](https://smith.langchain.com/)
- *(Only for `4_agent.py`)* A [Weatherstack](https://weatherstack.com/) API key

---

## Setup

### Using uv (recommended)

```bash
git clone <repo-url>
cd langsmith
uv sync                 # creates .venv and installs pinned dependencies from uv.lock
```

Run any script with:

```bash
uv run python 1_simple_llm_call.py
```

### Using pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .        # installs dependencies listed in pyproject.toml
```

---

## Environment Variables

Create a `.env` file in the project root. It is already listed in `.gitignore`.

```dotenv
# OpenAI
OPENAI_API_KEY=sk-...

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=default-project
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Authenticates `ChatOpenAI` and `OpenAIEmbeddings` |
| `LANGCHAIN_TRACING_V2` | Set to `true` to send traces to LangSmith |
| `LANGCHAIN_API_KEY` | Your LangSmith API key |
| `LANGCHAIN_ENDPOINT` | LangSmith API endpoint (EU accounts use `https://eu.api.smith.langchain.com`) |
| `LANGCHAIN_PROJECT` | Default LangSmith project. Most scripts override it in code (see [below](#langsmith-projects-created)) |

Every script calls `load_dotenv()`, so no extra wiring is needed. Once these are set, all LangChain/LangGraph runs are traced automatically.

---

## The Examples

### 1. Simple LLM Call: `1_simple_llm_call.py`

The smallest possible LCEL chain:

```
PromptTemplate("{question}") → ChatOpenAI() → StrOutputParser()
```

It asks *"What is the capital of Peru?"* and prints the answer. With tracing enabled, a single trace appears in LangSmith containing the prompt, model call and parser as child runs. This confirms your setup works.

```bash
uv run python 1_simple_llm_call.py
```

---

### 2. Sequential Chain: `2_sequential_chain.py`

A two-stage chain that passes the output of one model to a second model:

1. **`gpt-4o-mini`** (temperature 0.7) writes a detailed report on a topic (*"Unemployment in India"*).
2. **`gpt-4o`** (temperature 0.5) condenses that report into a 5-point summary.

```
prompt1 → gpt-4o-mini → parser → prompt2 → gpt-4o → parser
```

**LangSmith features shown:**
- Project set in code: `LANGCHAIN_PROJECT = "Sequential LLM App"`
- A custom `run_name` (`"Sequential Chain"`)
- `tags` (`llm app`, `report gen`, `summarization`) for filtering
- `metadata` that records which model was used at each stage

---

### 3. PDF RAG Chatbot (v1 → v4)

Four versions of the same Retrieval-Augmented Generation app over `islr.pdf`. The RAG logic is the same in all four; what changes between versions is **how the pipeline is traced** and, in v4, **how the index is cached**.

**Common pipeline:**

```
PyPDFLoader ─► RecursiveCharacterTextSplitter ─► OpenAIEmbeddings ─► FAISS
 (1 doc/page)    (chunk_size=1000, overlap=150)  (text-embedding-3-small)
                                                                        │
                                          retriever (similarity, k=4) ◄─┘
                                                        │
  { context: retriever | format_docs,  question: passthrough }
                                                        │
      ChatPromptTemplate ─► gpt-4o-mini (temp 0) ─► StrOutputParser
```

The system prompt tells the model to *answer only from the provided context* and to say "I don't know" otherwise. Each script prompts for one question on stdin and prints the answer.

| Version | What's new | LangSmith project |
|---|---|---|
| **`3_rag_v1.py`** | Baseline. Only the LCEL query chain is traced automatically; PDF loading, splitting and indexing are invisible in LangSmith. | `RAG Chatbot` |
| **`3_rag_v2.py`** | Wraps `load_pdf`, `split_documents` and `build_vectorstore` with `@traceable`, grouped under a `setup_pipeline` span. The query runs as `pdf_rag_query`. Setup and query appear as **two separate traces**. | `RAG Chatbot` |
| **`3_rag_v3.py`** | Adds a root `@traceable` function, `pdf_rag_full_run`, that runs both setup and query, so everything nests under **one trace**. Adds a `setup` tag. | from `.env` |
| **`3_rag_v4.py`** | Adds a **persistent FAISS index cache** (details below). Uses explicit `load_index` / `build_index` spans so cache hits and misses are visible in traces. Query run tagged `qa` with `metadata={"k": 4}`. | `Indexed Langchain APP` |

#### How the v4 index cache works

1. A cache key is computed as a SHA-256 hash over:
   - the PDF's content hash, size and modification time
   - `chunk_size`, `chunk_overlap`, and the embedding model name
   - a format version string
2. The index is stored in `.indices/<key>/` (`index.faiss`, `index.pkl`, and a readable `meta.json`).
3. If the directory exists, the index is loaded with `FAISS.load_local` (**no embedding cost**). Otherwise it is built, saved and traced as `build_index`.
4. Pass `force_rebuild=True` to `setup_pipeline_and_query()` to ignore the cache.

Resulting trace tree in LangSmith:

```
pdf_rag_full_run
├── setup_pipeline            [setup]
│   └── load_index | build_index   [index]
│         └── (build only) load_pdf → split_documents → build_vectorstore
└── pdf_rag_query             [qa]  {k: 4}
    ├── retriever
    ├── ChatPromptTemplate
    ├── ChatOpenAI
    └── StrOutputParser
```

```bash
uv run python 3_rag_v4.py
# PDF RAG ready. Ask a question (or Ctrl+C to exit).
# Q: What is the bias-variance trade-off?
```

> ⚠️ The index is loaded with `allow_dangerous_deserialization=True` because FAISS metadata is pickled. Only load indexes that you created yourself.

---

### 4. ReAct Agent with Tools: `4_agent.py`

A classic **ReAct** (Reason + Act) agent built with `create_react_agent` and `AgentExecutor`, using the standard `hwchase17/react` prompt pulled from LangChain Hub.

**Tools:**
| Tool | Description |
|---|---|
| `DuckDuckGoSearchRun` | Web search, no API key needed |
| `get_weather_data(city)` | Custom `@tool` that calls the Weatherstack API for current weather |

The executor runs with `verbose=True` and `max_iterations=5`. Example queries are included as comments in the script:

- *What is the current temp of gurgaon?* (single tool)
- *What is the release date of Dhadak 2?* (search)
- *Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.* (multi-step, chained tools)

In LangSmith (project `Langchain Agent App`) each Thought → Action → Observation loop appears as nested LLM and tool runs, which makes it easy to debug the agent's reasoning.

---

### 5. LangGraph Essay Evaluator: `5_langgraph.py`

A **LangGraph** workflow that grades a UPSC-style essay (*"India and AI Time"*, written with deliberately poor English) along three dimensions **in parallel**, then aggregates the results.

```
               ┌──► evaluate_language ──┐
               │                        │
    START ─────┼──► evaluate_analysis ──┼──► final_evaluation ──► END
               │                        │
               └──► evaluate_thought  ──┘
```

**Key concepts:**
- **Typed state** (`UPSCState` `TypedDict`) shared across nodes.
- **Reducer for parallel writes:** `individual_scores: Annotated[List[int], operator.add]` merges the score lists returned by the three parallel nodes instead of overwriting them.
- **Structured output:** `model.with_structured_output(EvaluationSchema)` returns a Pydantic object with `feedback: str` and `score: int` (0–10).
- **Aggregation:** `final_evaluation` summarizes the three feedback texts with the LLM and computes `avg_score`.

**LangSmith features shown:**
- Each node is also decorated with `@traceable`, with per-dimension `tags` (e.g. `dimension:language`) and `metadata`.
- The root run is named `evaluate_upsc_essay` and carries tags (`essay`, `langgraph`, `evaluation`) and metadata (essay length, model, dimensions).
- Project: `Trace Langgraph App`

**Output:** language, analysis, clarity and overall feedback, plus the individual scores and their average.

---

## LangSmith Tracing Techniques Used

| Technique | Where | How |
|---|---|---|
| Automatic tracing | All scripts | `LANGCHAIN_TRACING_V2=true` + API key in `.env` |
| Per-app projects | 2, 3 (v1/v2/v4), 4, 5 | `os.environ["LANGCHAIN_PROJECT"] = "..."` |
| Custom run names | 2, 3 (v2–v4), 5 | `chain.invoke(..., config={"run_name": ...})` |
| Tags & metadata | 2, 3 (v4), 5 | `config={"tags": [...], "metadata": {...}}` |
| Tracing plain Python functions | 3 (v2–v4), 5 | `@traceable(name=..., tags=..., metadata=...)` |
| Nested / hierarchical traces | 3 (v3, v4) | Calling traced functions inside a root `@traceable` function |
| Cache hit/miss visibility | 3 (v4) | Separate `load_index` vs `build_index` spans |
| Agent step inspection | 4 | Automatic tracing of `AgentExecutor` iterations |
| Graph node tracing | 5 | LangGraph auto-tracing + `@traceable` node functions |

---

## LangSmith Projects Created

After running every script, these projects will appear in your LangSmith workspace:

| Script | Project |
|---|---|
| `1_simple_llm_call.py` | value of `LANGCHAIN_PROJECT` in `.env` |
| `2_sequential_chain.py` | `Sequential LLM App` |
| `3_rag_v1.py`, `3_rag_v2.py` | `RAG Chatbot` |
| `3_rag_v3.py` | value of `LANGCHAIN_PROJECT` in `.env` |
| `3_rag_v4.py` | `Indexed Langchain APP` |
| `4_agent.py` | `Langchain Agent App` |
| `5_langgraph.py` | `Trace Langgraph App` |

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| `langchain` | 0.3.27 | Core framework, agents, text splitters |
| `langchain-core` | 0.3.74 | LCEL runnables, prompts, parsers |
| `langchain-openai` | 0.3.29 | `ChatOpenAI`, `OpenAIEmbeddings` |
| `langchain-community` | 0.3.27 | `PyPDFLoader`, `FAISS`, `DuckDuckGoSearchRun` |
| `langgraph` | 0.6.4 | Stateful graph workflows |
| `langsmith` | 0.4.13 | Tracing, `@traceable` |
| `faiss-cpu` | 1.11.0 | Vector similarity search |
| `pypdf` | 6.0.0 | PDF parsing |
| `duckduckgo-search` | 8.1.1 | Web search tool backend |
| `pydantic` | 2.11.7 | Structured output schemas |
| `python-dotenv` | 1.1.1 | Loading `.env` |

The full pinned dependency list is in `pyproject.toml` and `uv.lock`.

---

## Troubleshooting

- **No traces in LangSmith:** check that `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set in `.env`, and that you run the script from the project root so `.env` is found.
- **`FileNotFoundError: islr.pdf`:** run the RAG scripts from the repository root, or set `PDF_PATH` to an absolute path.
- **The first RAG run is slow or costs tokens:** embedding the whole PDF is expensive. Use `3_rag_v4.py`, which embeds once and reuses the cached index on later runs.
- **v4 rebuilds the index after cloning:** the cache key includes the PDF's modification time, which changes on checkout, so the committed `.indices/` entry may not match. The script will rebuild and cache a new index.
- **DuckDuckGo rate limits (`4_agent.py`):** the free search backend sometimes throttles requests. Wait and retry, or try a different query.
- **`hub.pull` fails:** pulling the ReAct prompt needs network access to LangChain Hub.
