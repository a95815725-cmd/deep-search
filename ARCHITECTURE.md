# Deep Search Agent — Architecture

---

## What This Is (and What It Isn't)

Traditional RAG does one thing: **search once, answer once.**

This agent does something fundamentally different — it **searches, reflects on what it found, decides if it's enough, and loops** until it is. That reflective loop is what makes it "deep search."

---

## The Agent Loop

```
                          ┌─────────────────────────────────────────────┐
                          │              AgentState (shared)            │
                          │  original_question   sub_queries            │
                          │  retrieved_documents (accumulates)          │
                          │  reflection_notes    gaps_identified         │
                          │  final_answer        citations               │
                          │  reasoning_trace     (accumulates)          │
                          └─────────────────────────────────────────────┘
                                              │
                                              ▼
                                      ┌─────────────┐
                                      │   PLANNER   │
                                      │             │
                                      │  Breaks the │
                                      │  question   │
                                      │  into 3–5   │
                                      │ sub-queries │
                                      └──────┬──────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │         SEARCHER          │
                              │                           │
                              │  For each pending query:  │
                              │  → ChromaDB vector search │
                              │
                              │  → Deduplicate via MD5    │
                              │  → Append to state        │
                              └──────────────┬────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │        REFLECTOR          │
                              │                           │
                              │  Reads retrieved chunks   │
                              │  Judges: is this enough?  │
                              │  If not → new sub-queries │
                              └──────────────┬────────────┘
                                             │
                              ┌──────────────┴────────────┐
                              │                           │
                    gaps + iter < 5             sufficient OR iter ≥ 5
                              │                           │
                              ▼                           ▼
                         (loop back)          ┌─────────────────────┐
                        to SEARCHER           │     SYNTHESIZER      │
                                              │                      │
                                              │  Ranks top-20 chunks │
                                              │  Writes grounded ans │
                                              │  Extracts citations  │
                                              │  Self-reports conf.  │
                                              └──────────┬───────────┘
                                                         │
                                                       END
                                              answer + citations
                                              + confidence score
                                              + reasoning trace
```

---

## Codebase Map

```
rabo-deep-search/
│
├── app.py                          Streamlit UI (3 tabs: Search / Compare / Docs)
├── mcp_server.py                   MCP server — exposes agent as tools for other agents
├── config.py                       Env / API key loader
│
├── src/
│   │
│   ├── agent/                      ◄── THE BRAIN
│   │   ├── state.py                AgentState TypedDict (the shared whiteboard)
│   │   ├── constants.py            MAX_ITERATIONS = 5
│   │   ├── graph.py                LangGraph StateGraph + run_deep_search()
│   │   ├── planner.py              Node 1: query decomposition
│   │   ├── searcher.py             Node 2: vector + web retrieval + dedup
│   │   ├── reflector.py            Node 3: gap detection + routing decision
│   │   ├── synthesizer.py          Node 4: grounded answer + citations
│   │   └── prompts/                All prompt templates (plain .txt files)
│   │       ├── planner_system.txt
│   │       ├── planner_human.txt
│   │       ├── reflector_system.txt
│   │       ├── reflector_human.txt
│   │       ├── synthesizer_system.txt
│   │       └── synthesizer_human.txt
│   │
│   ├── ingestion/                  ◄── THE PIPELINE
│   │   ├── pdf_parser.py           Section-aware PDF extraction (pdfplumber)
│   │   ├── chunker.py              Smart chunking — tables kept whole, metadata tagged
│   │   └── vector_store.py         ChromaDB wrapper (embed → store → search)
│   │
│   ├── models/
│   │   └── model_registry.py       LLM factory: OpenAI / Anthropic / Google
│   │
│   ├── evaluation/                 ◄── HOW WE KNOW IT WORKS
│   │   ├── metrics.py              detect_hallucinations + answer_completeness
│   │   ├── benchmark.py            15 hard financial questions (categorised)
│   │   ├── llm_judge.py            3 async LLM judges (parallel via asyncio.gather)
│   │   └── prompts/                Judge prompt templates (.json — system+human pairs)
│   │       ├── planner_judge.json
│   │       ├── reflector_judge.json
│   │       └── synthesizer_judge.json
│   │
│   └── tools/
│       └── web_search.py           Tavily API (optional live web context)
```

---

## Data Flow Through One Question

```
User question: "What were Rabobank's key risks in 2022?"
        │
        ▼
  run_deep_search()
  ┌─ initial AgentState ──────────────────────────────┐
  │  original_question = "What were Rabobank's..."    │
  │  sub_queries       = []                           │
  │  retrieved_docs    = []   ← accumulates via add() │
  │  reasoning_trace   = []   ← accumulates via add() │
  └───────────────────────────────────────────────────┘
        │
        ▼  PLANNER
  sub_queries = [
    "Rabobank risk factors 2022",
    "Rabobank credit risk annual report",
    "Rabobank liquidity risk 2022",
    "Rabobank regulatory capital requirements"
  ]
        │
        ▼  SEARCHER [iter 1]
  retrieved_docs += 18 unique chunks from ChromaDB

        │
        ▼  REFLECTOR [iter 1]
  reflection_notes   = "Found credit + liquidity. Missing capital ratios."
  gaps_identified    = ["CET1 ratio figures for 2022 not found"]
  sufficient_context = False  ──► loop back, new query: "Rabobank CET1 2022"

        │
        ▼  SEARCHER [iter 2]
  retrieved_docs += 7 more unique chunks

        │
        ▼  REFLECTOR [iter 2]
  sufficient_context = True   ──► proceed to synthesis

        │
        ▼  SYNTHESIZER
  answer           = "## Key Risks\n\nRabobank identified..."
  citations        = [{doc_name, page, section, excerpt}, ...]
  confidence_score = 0.88
  reasoning_trace  = [step 1, step 2, step 3, step 4, step 5]
```

---

## Evaluation Layer

> Reference: [Anthropic's demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
> We're a research agent — the article has specific guidance for this type.

The article defines three grader types: **code-based**, **model-based**, and **human**. We have the first two.

### Code-based graders (deterministic, cheap, reproducible)

```
detect_hallucinations(answer, retrieved_docs)
  ─────────────────────────────────────────────────────────────────
  → "Groundedness check" in the article's terminology.
  Extracts all numbers from the final answer, then checks whether each
  appears in any source document (with 1% rounding tolerance).
  Returns: hallucinated_numbers, verified_numbers, hallucination_rate.
  Why numbers: in financial research, hallucinated percentages, revenues,
  and dates are the most dangerous class of error.

calculate_answer_completeness(answer, ground_truth_hints)
  ─────────────────────────────────────────────────────────────────
  → "Coverage check" in the article's terminology.
  Checks what fraction of expected key facts/phrases appear in the answer.
  Returns 0.0–1.0. Benchmark questions include hint phrases as ground truth.
```

### Model-based graders (flexible, captures nuance)

```
asyncio.gather(                        ← all 3 fire in parallel
    judge_planner(sub_queries),        → "were these queries well-decomposed?"
    judge_reflector(gaps, verdict),    → "was the gap assessment accurate?"
    judge_synthesizer(answer, docs),   → "is the answer faithful to sources?"
)

Each returns:  score (1–5)  |  verdict (pass/partial/fail)  |  reasoning  |  issues

Per-node judges let us pinpoint WHERE a run degraded —
the planner, the reflector's gap detection, or the final synthesis —
rather than only knowing the end answer was wrong.
```

### Benchmark tasks

```
15 questions across 4 categories: financial_analysis, comparison,
trend_analysis, risk_assessment.

Difficulty: basic (5) / intermediate (7) / advanced (3).

The article recommends 20–50 tasks as a starting point. We're close.
Each question includes expected_answer_hints for completeness scoring.
```

### What we'd add at production scale

```
1. pass@k and pass^k — handling non-determinism
   ─────────────────────────────────────────────────────────────────
   Agent outputs vary between runs. The article defines two metrics:
     pass@k  — probability of at least 1 correct answer in k tries
               (relevant when one good answer matters, e.g. retrieval)
     pass^k  — probability ALL k trials pass
               (relevant for customer-facing reliability guarantees)
   We currently run one trial. At scale, run 3–5 and track both metrics.

2. Capability vs regression split
   ─────────────────────────────────────────────────────────────────
   The article distinguishes:
     capability evals — "can the agent do X?" (low pass rate → hill to climb)
     regression evals — "does it still do what it used to?" (target ~100%)
   Our 15 questions should be tagged with this distinction. Advanced
   questions are capability evals; basic ones should graduate to regression.

3. Trial isolation
   ─────────────────────────────────────────────────────────────────
   Each eval run should start from a clean vector store state to avoid
   correlated failures from shared state (the article calls this out
   explicitly — Claude once gained an unfair advantage by reading git
   history between trials).

4. Transcript analysis
   ─────────────────────────────────────────────────────────────────
   We already store reasoning_trace in AgentState. The article recommends
   actively reading transcripts: "When a task fails, the transcript tells
   you whether the agent made a genuine mistake or the graders rejected
   a valid solution." A UI panel to browse the full trace per run is the
   natural next step beyond the judge scores.
```

---

## Dependency Philosophy

```
LangGraph          ──► orchestration only (state graph, conditional routing)
                        "worth it for the loop — managing state + routing manually
                         is exactly what a state graph solves"

langchain_openai   ──► with_structured_output() + OpenAIEmbeddings
langchain_anthropic ─► with_structured_output()         "thin adapters:
langchain_google   ──► with_structured_output()          Pydantic → JSON schema
                        "only kept for schema-to-function-calling conversion"   → provider API"

OpenAI API         ──► all LLM calls (GPT-4o, GPT-4o Mini) ◄── DIRECT
Anthropic API      ──► all LLM calls (Claude 3.5/3.7)       ◄── DIRECT
Google API         ──► all LLM calls (Gemini 2.0)            ◄── DIRECT
ChromaDB           ──► vector store                          ◄── DIRECT
Tavily API         ──► web search                            ◄── DIRECT
pdfplumber         ──► PDF parsing                           ◄── DIRECT
```

---

## Why Deep Search vs RAG

| | Traditional RAG | This Agent |
|---|---|---|
| Search | Once, fixed query | Iterative, adaptive |
| Query | User's raw question | Decomposed into targeted sub-queries |
| Self-critique | None | Reflector judges its own gaps |
| Loops | No | Up to 5 search-reflect cycles |
| Traceability | None | Citations + full reasoning trace |
| Evaluation | Manual | LLM-as-judge per node (async) |
| Extensibility | Monolithic | Exposed as MCP tools → any agent can call it |

---

## MCP Surface (Agent-to-Agent)

> Informed by [Anthropic's code execution with MCP post](https://www.anthropic.com/engineering/code-execution-with-mcp):
> the primary MCP scaling problem is *"tool definitions overload the context window"* when agents
> connect to hundreds of tools. Our counter: keep the surface deliberately minimal.

```
mcp_server.py exposes the agent as callable tools:

  list_knowledge_base()                 ← discover what's indexed first
  search_docs(query, top_k,             ← raw vector search
              summary_only=false)         summary_only=true returns only metadata,
                                          no chunk content (saves ~80% of tokens
                                          when just exploring before deep_search)
  deep_search(question, model)          ← full agent loop
  ingest_document(path)                 ← add PDF to knowledge base

  Resource: docs://corpus               ← full corpus metadata (MCP resource)
```

### Design decisions mapped to the article

```
1. Minimal tool surface (4 tools, no overlap)
   ─────────────────────────────────────────────────────────────────
   The article warns: "if a human engineer can't definitively say which
   tool to use, an AI agent can't be expected to do better."
   Each tool here has a single, non-overlapping job.

2. Progressive disclosure via list_knowledge_base first
   ─────────────────────────────────────────────────────────────────
   The intended calling pattern mirrors the article's "just-in-time" model:
     list_knowledge_base()            → what years/docs are available?
     search_docs(..., summary_only=true) → which docs are relevant? (metadata only)
     deep_search(...)                 → now run the full agent loop
   Each step loads only as much context as needed for the current decision.

3. summary_only on search_docs — token-efficient tool results
   ─────────────────────────────────────────────────────────────────
   The article: "agents can filter and transform results in code before
   returning them... the agent sees five rows instead of 10,000."
   summary_only=true returns doc/page/score only — no chunk content.
   A calling agent scouting 5 documents saves ~2,500 tokens vs full results.

4. Lazy singleton initialisation
   ─────────────────────────────────────────────────────────────────
   The vector store and graph are only initialised on the first tool call.
   Startup is near-instant; cost is paid only when a tool is actually used.

5. SKILL.md validates this architecture
   ─────────────────────────────────────────────────────────────────
   The article explicitly says: "Adding a SKILL.md file to saved functions
   creates a structured skill that models can reference and use."
   Our .cursor/skills/deep-search-agent/SKILL.md does exactly this.
```

### Production evolution (what code execution with MCP would add)

```
At scale (hundreds of tools across multiple MCP servers), the next step
would be presenting tools as a navigable file tree rather than loading all
definitions upfront. The article shows a 98.7% token reduction for large
tool sets using this approach. For our 4-tool server it's unnecessary —
but the foundation is the same deliberate surface design.
```

---

## Context Engineering

> Inspired by [Anthropic's effective context engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):
> *"Find the smallest possible set of high-signal tokens that maximise the likelihood of your desired outcome."*

One of the core problems with multi-iteration agents is **context rot** — by iteration 3 or 4, the LLM is seeing stale, repeated, or irrelevant chunks that dilute the signal. Every design decision below is a direct answer to that problem.

### How context is kept intact across iterations

```
AgentState uses operator.add on two fields:

  retrieved_documents: Annotated[List[Dict], operator.add]
  reasoning_trace:     Annotated[List[str],  operator.add]

operator.add means LangGraph *appends* to these lists on every node return —
it never overwrites them. Chunks found in iteration 1 are still available
to the synthesizer in iteration 5.
```

### Deduplication — no chunk enters twice

```
Searcher computes an MD5 hash of each retrieved chunk:

  hash = MD5( content[:500] | source | page )

Hashes are tracked in a set across all iterations.
If the same chunk comes back in iteration 1 and iteration 3,
it is silently dropped — the LLM never sees it twice.

Why matters: without this, a highly similar query in iteration 2
would pull the same 6 chunks, wasting ~20% of the context budget
on content the reflector already saw.
```

### Dynamic context budget by model

```
Each node's character budget is scaled to the active model's context window:

  context_budget(model_name, base_chars) = base_chars × min(window / 128k, 4.0)

  Model             Context Window    Reflector Budget    Synthesizer Budget
  ──────────────────────────────────────────────────────────────────────────
  gpt-4o            128,000           12,000 chars        18,000 chars  (1×)
  claude-3.7-sonnet 200,000           18,750 chars        28,125 chars  (1.56×)
  gemini-2.0-flash  1,000,000         48,000 chars        72,000 chars  (4× cap)

This avoids the waste of using a 12k budget on a model that can handle 200k.
The cap at 4× prevents the budget from growing unboundedly on very large windows
where token-attention quality starts to degrade anyway (context rot).
```

### Different context budgets per node — intentional

```
                      MAX_CHUNK_CHARS    Base Context Budget    Top-N docs
  ──────────────────────────────────────────────────────────────────────
  Reflector              800 chars          12,000 chars          all (sorted)
  Synthesizer          1,000 chars          18,000 chars          top 20

Reflector gets a *narrower* view: shorter chunks, tighter budget.
It only needs to judge whether gaps exist — it doesn't write the answer.
Giving it the full context would dilute its gap-detection signal.

Synthesizer gets a *richer* view: longer chunks, larger budget, ranked docs.
It needs enough verbatim text to ground every claim and extract citations.
```

### Ranking before truncation

```
When the synthesizer fills its context window, it doesn't take chunks
in retrieval order. It sorts first:

  key = (source_priority, -score)

  source_priority:  0 = vector_store,  1 = web
  score:            cosine similarity (higher = more relevant)

Vector store results rank above web results.
Within each source, higher-confidence chunks rank first.
Only then does the 18k char budget cut off the tail.

This means the LLM always sees the most relevant, most trusted content —
not just the most recently retrieved.
```

### Tool result clearing

```
Anthropic's engineering team calls this "one of the safest lightest-touch forms
of compaction."

Problem: SubQuery.results stores raw retrieved chunks for each search call.
After the reflector runs, those chunks are already merged into retrieved_documents.
Keeping them means every subsequent reflector pass re-reads the same raw data from
two places, burning context tokens for zero new signal.

Solution: At the end of every reflector node, results are cleared from all
already-searched sub-queries before they are written back to state:

  cleared_sub_queries = [
      SubQuery(query=sq.query, status=sq.status, results=[])  # ← cleared
      for sq in state.sub_queries
  ]

This keeps sub_queries as a lightweight index (query + status) and lets
retrieved_documents be the single source of truth for context content.
```

### What we still want to experiment with

```
1. Semantic deduplication vs hash deduplication
   ─────────────────────────────────────────────────────────────────
   Current: MD5 hash on raw content — exact duplicate detection only.
   Problem: two chunks with 90% overlapping text both pass through.
   Experiment: embed each chunk and reject any with cosine similarity
   > 0.95 to an already-seen chunk. Cuts near-duplicate noise further.

2. Compression before synthesis
   ─────────────────────────────────────────────────────────────────
   Opportunity: before the synthesizer sees the full context budget,
   run a cheap "compress" step — a small model that rewrites each
   retrieved chunk into a 1-sentence fact summary, reducing context
   volume while preserving the key claims. Trades one cheap LLM call
   for a much cleaner synthesis context.

3. Per-iteration context windowing for the reflector
   ─────────────────────────────────────────────────────────────────
   Current: the reflector sees ALL accumulated chunks every iteration.
   Alternative: give the reflector only the *new* chunks from this
   iteration + a summary of previous iterations. Keeps the reflector's
   focus on what just arrived, not re-reading already-assessed content.
```
