# Financial Deep Search Agent

> A proof-of-concept demonstrating **deep research capabilities on financial documents** using agentic AI — going beyond traditional RAG with iterative retrieval, reflection, and multi-model evaluation.

---

## What This Is

Traditional Retrieval-Augmented Generation (RAG) for financial documents is a single-pass system: embed a question, fetch the top-*k* chunks, hand them to an LLM, and return an answer. This works reasonably well for simple factual lookups, but fails badly on the kind of questions that actually matter in finance — questions that span multiple sections of a 10-K, require comparing figures across years, or need the model to notice when a risk factor in the footnotes contradicts an optimistic management narrative.

This project implements a **LangGraph-powered agentic loop** that mimics how a skilled financial analyst actually reads a document: first decomposing the question into targeted sub-queries, then searching iteratively, then stopping to reflect on what is still missing, and finally synthesising a grounded answer with precise citations. The agent can loop up to five times before producing its answer, meaning it will go back and re-search if it detects gaps — something a single-pass RAG system simply cannot do.

The application also provides a **multi-model evaluation layer**: you can run the same question through GPT-4o, Claude 3.5 Sonnet, Claude 3.7 Sonnet, and Gemini 2.0 Flash simultaneously and compare confidence scores, hallucination risk, iteration counts, and answer quality side by side. A built-in benchmark suite of 15 hard financial questions provides a repeatable standard for measuring model and system performance.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FINANCIAL DEEP SEARCH AGENT               │
│                        (LangGraph DAG)                       │
└─────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  INGEST  │    │  PLANNER │    │ SEARCHER │    │REFLECTOR │
  │          │    │          │    │          │    │          │
  │ PDF      │───▶│ Decompose│───▶│ Vector   │───▶│ Gap      │
  │ Parser   │    │ question │    │ search   │    │ detection│
  │          │    │ into N   │    │ per sub- │    │ & notes  │
  │ Chunker  │    │ sub-     │    │ query    │    │          │
  │          │    │ queries  │    │          │    │          │
  │ Vector   │    └──────────┘    └──────────┘    └────┬─────┘
  │ Store    │                                         │
  └──────────┘                           ┌─────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │      SUFFICIENT CONTEXT?     │
                          └──────────┬──────────┬────────┘
                                     │ YES       │ NO (max 5x)
                                     ▼           ▼
                              ┌────────────┐  ┌──────────────┐
                              │SYNTHESIZER │  │ Refine query │
                              │            │  │ & loop back  │
                              │ Final ans  │  │ to SEARCHER  │
                              │ + citations│  └──────────────┘
                              │ + confidence│
                              └────────────┘

Ingestion Pipeline:
  PDF → pdfplumber → section-aware chunker → tiktoken splitter → ChromaDB
```

---

## Key Features

| Feature | Description |
|---|---|
| **Query Decomposition** | Complex financial questions are broken into 3–5 targeted sub-queries by the planner node |
| **Iterative Retrieval** | The agent loops up to 5 times, re-searching with refined queries when gaps are detected |
| **Reflection & Gap Detection** | After each retrieval pass the reflector node identifies missing context and generates follow-up queries |
| **Multi-Model Support** | GPT-4o, Claude 3.5 Sonnet, Claude 3.7 Sonnet, and Gemini 2.0 Flash — swappable at runtime |
| **Financial-Aware Chunking** | Section-aware parsing preserves tables, footnotes, and accounting line items with surrounding context |
| **Confidence Scoring** | Each answer includes a 0–1 confidence score derived from source coverage and reflection notes |
| **Benchmark Suite** | 15 hard financial questions across revenue, margins, cash flow, risk, and capital allocation |
| **Hallucination Detection** | Numerical claims in the answer are verified against the cited source excerpts |
| **Streamlit UI** | Full 4-tab web interface: Deep Search, Model Comparison, Benchmark, Document Management |

---

## Why Deep Search vs Traditional RAG

| Dimension | Traditional RAG | Deep Search Agent |
|---|---|---|
| **Retrieval passes** | Single pass | Up to 5 iterative passes |
| **Query planning** | Raw question sent as-is | Decomposed into targeted sub-queries |
| **Gap handling** | None — returns whatever it found | Detects gaps, generates follow-up queries |
| **Context sufficiency** | No check | Explicit sufficiency gate before synthesis |
| **Multi-hop questions** | Poor (requires all info in top-k chunks) | Handles naturally via iteration |
| **Table / numeric grounding** | Weak | Section-aware chunking preserves table context |
| **Hallucination risk** | High on numeric claims | Reduced by source verification |
| **Answer traceability** | Vague citations | Step-by-step reasoning trace + precise citations |
| **Multi-model evaluation** | Typically single model | Native side-by-side comparison |

---

## Setup

### Prerequisites

- Python **3.11+**
- At least one API key: OpenAI (required for GPT-4o), Anthropic (for Claude), or Google (for Gemini)
- Tavily API key (optional, for web-augmented search)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/rabo-deep-search.git
cd rabo-deep-search

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

Open `.env` and fill in your keys:

```dotenv
# Required for GPT-4o
OPENAI_API_KEY=sk-...

# Optional — enables Claude models
ANTHROPIC_API_KEY=sk-ant-...

# Optional — enables Gemini models
GOOGLE_API_KEY=AIza...

# Optional — enables web-augmented search
TAVILY_API_KEY=tvly-...
```

### Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## Usage

### Adding Financial Documents

1. Open the **Document Management** tab in the sidebar.
2. Drag and drop one or more PDF files into the upload area. Supported types: 10-K filings, annual reports, earnings call transcripts.
3. Click **Ingest Documents**. The app will:
   - Parse each PDF with `pdfplumber`, preserving tables and section headers
   - Split into semantically coherent chunks using the financial-aware chunker
   - Embed and index chunks into ChromaDB
4. Once ingestion completes you will see document names, chunk counts, and detected year ranges in the document table.

> **Tip:** For time-series questions (e.g. "how did margins change over 3 years?"), upload multiple years of the same company's reports.

### Running a Deep Search

1. Go to the **Deep Search** tab.
2. Type your question in the text box. More specific questions with explicit scope (company, metric, time period) get better answers.
3. Select the model from the dropdown.
4. Click **Search**. A live reasoning trace will show each agent step as it executes.
5. Review the answer, confidence score badge (green ≥ 80%, yellow 50–80%, red < 50%), and cited source excerpts.
6. Expand **Reasoning Trace** to see the full agent decision log.
7. Use **Export as Markdown** to save the result.

**Example question:**
```
How did Apple's gross margin evolve between 2021 and 2023, and what were
the primary drivers cited by management? Were there any contradicting signals
in the risk factors section?
```

### Running Model Comparisons

1. Go to **Model Comparison**.
2. Enter your question and select 2–4 models with the multi-select.
3. Click **Compare Models**. Each model runs sequentially.
4. Results appear side by side with a comparison table and two charts: a confidence bar chart and a multi-metric radar chart.

### Running Benchmarks

1. Go to **Benchmark**.
2. Optionally filter the 15 built-in questions by category or difficulty.
3. Select one or more models, then click **Run Benchmark**.
4. A progress bar tracks each question/model pair. Results populate a table showing score, confidence, iterations, and time.
5. Summary tables show average score per model and best model per category.
6. Export results as CSV for offline analysis.

---

## Project Structure

```
rabo-deep-search/
│
├── app.py                          # Streamlit UI (4 tabs)
├── config.py                       # Settings, env var loading
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for API keys
├── README.md                       # This file
│
├── data/
│   └── temp_uploads/               # Transient upload staging (auto-created)
│
└── src/
    ├── __init__.py
    │
    ├── agent/
    │   ├── __init__.py
    │   ├── state.py                # AgentState TypedDict (LangGraph)
    │   └── graph.py                # LangGraph DAG definition + run_deep_search()
    │
    ├── ingestion/
    │   ├── __init__.py
    │   ├── pdf_parser.py           # FinancialPDFParser — pdfplumber-based
    │   ├── chunker.py              # FinancialChunker — section-aware splitting
    │   └── vector_store.py         # FinancialVectorStore — ChromaDB wrapper
    │
    ├── models/
    │   ├── __init__.py
    │   └── model_registry.py       # AVAILABLE_MODELS list + factory function
    │
    ├── tools/
    │   └── __init__.py             # LangChain tools used by the agent
    │
    └── evaluation/
        ├── __init__.py
        ├── benchmark.py            # BENCHMARK_QUESTIONS + BenchmarkRunner
        └── metrics.py              # detect_hallucinations(), calculate_answer_completeness()
```

---

## Example Questions This System Can Answer

These questions are deliberately hard — they require multi-section reasoning, cross-year comparison, or careful reading of footnotes and risk disclosures:

1. **Cross-year margins:** *"How did the gross margin trend from 2020–2023 and what specific cost items drove the inflection in 2022?"*

2. **Segment analysis:** *"Break down revenue and operating income by segment for the last two years. Which segment's margins deteriorated fastest, and what did management attribute this to?"*

3. **Cash conversion:** *"How did free cash flow conversion (FCF / net income) trend over 3 years, and what balance-sheet movements explain the divergence from GAAP earnings?"*

4. **Risk vs narrative:** *"Management highlighted AI as a growth opportunity in the letter to shareholders. What specific risks related to AI were disclosed in Item 1A, and are there any internal contradictions?"*

5. **Capital allocation:** *"How much capital was returned to shareholders via buybacks and dividends versus deployed into capex and acquisitions over the past 3 fiscal years? What does this imply about management's capital allocation priorities?"*

---

## Benchmark Results

*Run benchmarks via the UI and export CSV to populate this table.*

| Model | Avg Score | Avg Confidence | Avg Iterations | Avg Time (s) | Questions Run |
|---|---|---|---|---|---|
| GPT-4o | — | — | — | — | — |
| Claude 3.5 Sonnet | — | — | — | — | — |
| Claude 3.7 Sonnet | — | — | — | — | — |
| Gemini 2.0 Flash | — | — | — | — | — |

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| **Agent orchestration** | LangGraph 0.3 | Stateful DAG with conditional edges; supports streaming |
| **LLM providers** | LangChain (OpenAI, Anthropic, Google) | Unified interface; easy model swapping |
| **Vector store** | ChromaDB | Lightweight, file-based, no infra required for POC |
| **PDF parsing** | pdfplumber + pypdf | Reliable table extraction; preserves layout metadata |
| **Tokenisation** | tiktoken | Accurate chunk-size control for OpenAI models |
| **UI** | Streamlit 1.42 | Rapid iteration; built-in state management |
| **Visualisation** | Plotly 6.0 | Interactive charts within Streamlit |
| **Data** | Pandas + NumPy | Benchmark aggregation and result tables |
| **Web search** | Tavily | Optional augmentation for web-available filings |
| **Configuration** | python-dotenv + Pydantic | Type-safe settings from `.env` |

---

## Contributing

This is a POC. To extend it:

- **Add a new LLM**: register it in `src/models/model_registry.py`
- **Add a new benchmark question**: append to `BENCHMARK_QUESTIONS` in `src/evaluation/benchmark.py`
- **Improve chunking**: edit `src/ingestion/chunker.py` — the chunker uses section headers and a sliding window by default
- **Add a new agent node**: add a node function to `src/agent/graph.py` and wire it into the `StateGraph`

---

## License

MIT — see `LICENSE` for details.
