"""
Financial Deep Search Agent — Streamlit Application
"""

import asyncio
import os
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Financial Deep Search Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
h1, h2, h3, h4 { color: #f0c040 !important; }
p, li, span, label { color: #c9d1d9; }
[data-testid="stTabs"] button { color: #8b949e !important; font-weight: 600; font-size: 14px; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #f0c040 !important; border-bottom: 2px solid #f0c040; background: #1c2128; }
.stButton > button { background: linear-gradient(135deg,#1a3a5c,#1f4f7a); color: #f0c040 !important; border: 1px solid #2d5f8a; border-radius: 8px; font-weight: 700; padding: 0.5rem 1.5rem; }
.stButton > button:hover { background: linear-gradient(135deg,#1f4f7a,#256aa0); border-color: #f0c040; }
.answer-card { background: #1c2128; border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.4); }
.citation-card { background: #161b22; border: 1px solid #21262d; border-left: 3px solid #f0c040; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
.metric-card { background: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 1rem; text-align: center; }
.step-item { background: #161b22; border-left: 3px solid #1f6feb; border-radius: 6px; padding: 0.6rem 1rem; margin: 0.3rem 0; font-size: 0.85rem; color: #8b949e; }
.badge-green  { background:#1a4a2a; color:#3fb950; border:1px solid #238636; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:700; }
.badge-yellow { background:#3d2e00; color:#f0c040; border:1px solid #9e6a03; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:700; }
.badge-red    { background:#4a1a1a; color:#f85149; border:1px solid #6e1a1a; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:700; }
.dot-green { display:inline-block; width:10px; height:10px; background:#3fb950; border-radius:50%; margin-right:6px; }
.dot-red   { display:inline-block; width:10px; height:10px; background:#f85149; border-radius:50%; margin-right:6px; }
hr { border-color: #30363d !important; }
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { background: #1c2128 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; }
[data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div { background: #1c2128 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
[data-testid="stFileUploader"] { background: #1c2128; border: 1px dashed #30363d; border-radius: 10px; padding: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Lazy module loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_modules():
    errors, modules = {}, {}
    pairs = [
        ("FinancialPDFParser", "src.ingestion.pdf_parser", "FinancialPDFParser"),
        ("FinancialChunker", "src.ingestion.chunker", "FinancialChunker"),
        ("FinancialVectorStore", "src.ingestion.vector_store", "FinancialVectorStore"),
        ("run_deep_search", "src.agent.graph", "run_deep_search"),
        ("deep_search_graph", "src.agent.graph", "deep_search_graph"),
        ("AVAILABLE_MODELS", "src.models.model_registry", "AVAILABLE_MODELS"),
        ("list_available_models", "src.models.model_registry", "list_available_models"),
        ("is_model_available", "src.models.model_registry", "is_model_available"),
        ("detect_hallucinations", "src.evaluation.metrics", "detect_hallucinations"),
        ("run_all_judges", "src.evaluation.llm_judge", "run_all_judges"),
        ("settings", "config", "settings"),
    ]
    for key, mod_path, attr in pairs:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            modules[key] = getattr(mod, attr)
        except Exception as e:
            errors[key] = str(e)
    return modules, errors


MODULES, MODULE_ERRORS = load_modules()


def get_mod(k):
    return MODULES.get(k)


# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {
    "vector_store": None,
    "ingested_docs": [],
    "last_search_result": None,
    "last_comparison_results": {},
    "confirm_clear": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ──────────────────────────────────────────────────────────────────
def confidence_badge(score: float) -> str:
    if score >= 0.8:
        return f'<span class="badge-green">✓ {score:.0%}</span>'
    elif score >= 0.5:
        return f'<span class="badge-yellow">~ {score:.0%}</span>'
    return f'<span class="badge-red">⚠ {score:.0%}</span>'


def get_configured_models() -> list[dict]:
    """Return only models whose API key is configured, for use in dropdowns."""
    AVAILABLE_MODELS = get_mod("AVAILABLE_MODELS")
    is_available = get_mod("is_model_available")
    if not AVAILABLE_MODELS:
        return [{"id": "gpt-4o", "name": "GPT-4o"}]
    result = []
    for model_id, info in AVAILABLE_MODELS.items():
        configured = is_available(model_id) if is_available else False
        if configured:
            result.append({"id": model_id, "name": info.get("display_name", model_id)})
    return result or [{"id": "gpt-4o", "name": "GPT-4o"}]


def get_all_models() -> list[dict]:
    """Return all models with a configured flag — for display purposes."""
    AVAILABLE_MODELS = get_mod("AVAILABLE_MODELS")
    is_available = get_mod("is_model_available")
    if not AVAILABLE_MODELS:
        return []
    return [
        {
            "id": mid,
            "name": info.get("display_name", mid),
            "provider": info.get("provider", ""),
            "configured": is_available(mid) if is_available else False,
        }
        for mid, info in AVAILABLE_MODELS.items()
    ]


def result_to_markdown(result: dict, question: str) -> str:
    lines = [
        "# Deep Search Result",
        "",
        f"**Question:** {question}",
        f"**Model:** {result.get('model_name', 'N/A')}",
        f"**Confidence:** {result.get('confidence_score', 0):.1%}",
        f"**Iterations:** {result.get('iteration_count', 0)}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Answer",
        "",
        result.get("final_answer", ""),
        "",
        "---",
        "",
        "## Citations",
        "",
    ]
    for i, c in enumerate(result.get("citations", []), 1):
        lines += [
            f"**[{i}] {c.get('doc_name', '?')}**  ",
            f"Section: {c.get('section', 'N/A')} | Page: {c.get('page_num', 'N/A')}",
            f"> {c.get('text_excerpt', '')}",
            "",
        ]
    return "\n".join(lines)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-size:2.4rem">🔍</div>
        <h2 style="font-size:1.1rem;margin:0;line-height:1.3">Financial<br>Deep Search Agent</h2>
        <p style="color:#8b949e;font-size:0.75rem;margin:0">Agentic RAG · Multi-Model · LangGraph</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Loaded Documents**")
    docs = st.session_state.ingested_docs
    if docs:
        st.markdown(f"📄 **{len(docs)}** document(s)")
        st.markdown(f"🧩 **{sum(d.get('chunks', 0) for d in docs):,}** chunks indexed")
        for d in docs[:5]:
            name = d["name"]
            st.markdown(f"  • {name[:28]}{'…' if len(name) > 28 else ''}")
        if len(docs) > 5:
            st.markdown(f"  *+{len(docs) - 5} more…*")
    else:
        st.markdown("*No documents loaded*")
        st.caption("Go to Document Management ↗")

    st.divider()
    with st.expander("⚙ Module Status", expanded=False):
        if not MODULE_ERRORS:
            st.success("All modules loaded ✓")
        else:
            st.warning(f"{len(MODULE_ERRORS)} module(s) unavailable")
            for mod, err in MODULE_ERRORS.items():
                st.markdown(f"**{mod}**: `{err[:80]}`")
    st.caption("v0.1.0 · Built with LangGraph + Streamlit")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_search, tab_compare, tab_docs = st.tabs(
    [
        "🔍  Deep Search",
        "⚖️  Model Comparison",
        "📁  Document Management",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEEP SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("## 🔍 Deep Search")
    st.markdown(
        "Ask complex questions. The agent decomposes, searches iteratively, reflects on gaps, and synthesises a grounded answer."
    )
    st.divider()

    configured_models = get_configured_models()
    model_options = {m["name"]: m["id"] for m in configured_models}

    col_q, col_m = st.columns([3, 1])
    with col_q:
        question = st.text_area(
            "Question",
            placeholder="e.g. What are the key risks and how do they relate to the company's capital position?",
            height=110,
            key="search_question",
        )
    with col_m:
        selected_model_name = st.selectbox("Model", list(model_options.keys()), key="search_model")
        search_btn = st.button("🔍  Search", use_container_width=True, key="btn_search")

    if search_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        elif st.session_state.vector_store is None:
            st.error("No documents loaded. Go to **Document Management** and upload a PDF first.")
        else:
            selected_model_id = model_options[selected_model_name]
            run_deep_search = get_mod("run_deep_search")
            deep_search_graph = get_mod("deep_search_graph")

            if run_deep_search is None:
                st.error(f"Agent unavailable: {MODULE_ERRORS.get('run_deep_search', '')}")
            else:
                with st.container():
                    st.markdown("### ⏳ Searching…")
                    status_box = st.status("Running deep search agent…", expanded=True)
                    with status_box:
                        step_area = st.empty()
                        steps_so_far = []

                        def _render_steps():
                            step_area.markdown(
                                "\n".join(
                                    f'<div class="step-item">▶ {s}</div>' for s in steps_so_far[-8:]
                                ),
                                unsafe_allow_html=True,
                            )

                        try:
                            t0 = time.time()
                            result = None

                            if deep_search_graph is not None:
                                inputs = {
                                    "original_question": question.strip(),
                                    "model_name": selected_model_id,
                                    "sub_queries": [],
                                    "search_strategy": "",
                                    "retrieved_documents": [],
                                    "iteration_count": 0,
                                    "reflection_notes": "",
                                    "gaps_identified": [],
                                    "sufficient_context": False,
                                    "final_answer": "",
                                    "citations": [],
                                    "confidence_score": 0.0,
                                    "reasoning_trace": [],
                                }
                                config = {
                                    "configurable": {"vector_store": st.session_state.vector_store}
                                }
                                for event in deep_search_graph.stream(
                                    inputs, config=config, stream_mode="updates"
                                ):
                                    for _, node_state in event.items():
                                        for t in node_state.get("reasoning_trace", []):
                                            if t not in steps_so_far:
                                                steps_so_far.append(t)
                                        _render_steps()
                                        result = node_state
                            else:
                                result = run_deep_search(
                                    question=question.strip(),
                                    model_name=selected_model_id,
                                    vector_store=st.session_state.vector_store,
                                )

                            elapsed = time.time() - t0
                            result["_elapsed"] = elapsed
                            result["_model_display"] = selected_model_name
                            st.session_state.last_search_result = result
                            status_box.update(
                                label=f"✅ Completed in {elapsed:.1f}s", state="complete"
                            )

                        except Exception as exc:
                            st.error(f"Search failed: {exc}")

    result = st.session_state.last_search_result
    if result:
        st.divider()
        confidence = result.get("confidence_score", 0.0)
        iterations = result.get("iteration_count", 0)
        n_citations = len(result.get("citations", []))
        elapsed = result.get("_elapsed", 0)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, colour in [
            (c1, f"{confidence:.0%}", "Confidence", "#f0c040"),
            (c2, str(iterations), "Iterations", "#58a6ff"),
            (c3, str(n_citations), "Citations", "#3fb950"),
            (c4, f"{elapsed:.1f}s", "Time", "#bc8cff"),
        ]:
            with col:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div style="font-size:1.8rem;font-weight:700;color:{colour}">{val}</div>
                    <div style="color:#8b949e;font-size:0.8rem">{label}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
        <div class="answer-card">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem">
                <span style="font-weight:700;color:#f0c040">Answer — {result.get("_model_display", "")}</span>
                {confidence_badge(confidence)}
            </div>
        </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(result.get("final_answer", "*No answer generated.*"))

        citations = result.get("citations", [])
        if citations:
            st.markdown("#### 📎 Citations")
            for i, c in enumerate(citations, 1):
                with st.expander(
                    f"[{i}] {c.get('doc_name', '?')} — {c.get('section', '')}, p.{c.get('page_num', '')}"
                ):
                    st.markdown(
                        f"""
                    <div class="citation-card">
                        <p style="color:#8b949e;font-size:0.8rem;margin-bottom:0.4rem">
                            📄 <b>{c.get("doc_name", "")}</b> &nbsp;|&nbsp;
                            Section: <b>{c.get("section", "N/A")}</b> &nbsp;|&nbsp;
                            Page: <b>{c.get("page_num", "N/A")}</b>
                        </p>
                        <p style="color:#c9d1d9;font-style:italic">"{c.get("text_excerpt", "")}"</p>
                    </div>""",
                        unsafe_allow_html=True,
                    )

        trace = result.get("reasoning_trace", [])
        if trace:
            with st.expander("🧠 Reasoning Trace", expanded=False):
                for i, step in enumerate(trace, 1):
                    st.markdown(
                        f'<div class="step-item"><b style="color:#f0c040">Step {i}</b>  {step}</div>',
                        unsafe_allow_html=True,
                    )

        # ── LLM-as-Judge panel ────────────────────────────────────────────────
        run_all_judges = get_mod("run_all_judges")
        if run_all_judges:
            st.markdown("#### 🧑‍⚖️ LLM Evaluation")
            if st.button("Run Judges (parallel)", key="run_judges"):
                with st.spinner("Running 3 judges in parallel…"):
                    try:
                        judge_scores = asyncio.run(run_all_judges(result))
                        st.session_state["judge_scores"] = judge_scores
                    except Exception as exc:
                        st.error(f"Judge run failed: {exc}")

            if "judge_scores" in st.session_state:
                scores = st.session_state["judge_scores"]
                overall = scores.get("overall_score", 0.0)

                # Overall score bar
                colour = "#3fb950" if overall >= 4 else "#f0c040" if overall >= 3 else "#f85149"
                st.markdown(
                    f"""
                <div class="metric-card" style="margin-bottom:1rem">
                    <div style="font-size:1.6rem;font-weight:700;color:{colour}">{overall:.1f} / 5.0</div>
                    <div style="color:#8b949e;font-size:0.8rem">Overall Judge Score</div>
                </div>""",
                    unsafe_allow_html=True,
                )

                # Per-node scores
                jc1, jc2, jc3 = st.columns(3)
                for col, key, label in [
                    (jc1, "planner", "Planner"),
                    (jc2, "reflector", "Reflector"),
                    (jc3, "synthesizer", "Synthesizer"),
                ]:
                    node = scores.get(key)
                    if not node:
                        continue
                    score = node.score
                    verdict = node.verdict
                    node_colour = (
                        "#3fb950" if score >= 4 else "#f0c040" if score == 3 else "#f85149"
                    )
                    badge_cls = (
                        "badge-green"
                        if score >= 4
                        else "badge-yellow"
                        if score == 3
                        else "badge-red"
                    )
                    with col:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div style="font-size:1.4rem;font-weight:700;color:{node_colour}">{score}/5</div>
                            <div style="color:#8b949e;font-size:0.75rem;margin-bottom:0.3rem">{label}</div>
                            <span class="{badge_cls}">{verdict.upper()}</span>
                        </div>""",
                            unsafe_allow_html=True,
                        )
                        with st.expander("Details", expanded=False):
                            st.markdown(f"**Reasoning:** {node.reasoning}")
                            if node.issues:
                                st.markdown("**Issues:**")
                                for issue in node.issues:
                                    st.markdown(f"- {issue}")

        st.download_button(
            "⬇️  Export as Markdown",
            result_to_markdown(result, st.session_state.get("search_question", "")),
            file_name=f"deep_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## ⚖️ Model Comparison")
    st.markdown("Run the same question across multiple models and compare results side-by-side.")
    st.divider()

    configured_models = get_configured_models()
    all_models = get_all_models()
    model_options_full = {m["name"]: m["id"] for m in configured_models}

    if not configured_models:
        st.warning("No models configured. Add API keys to `.env` and restart.")
    else:
        # Show available vs unavailable
        unconfigured = [m for m in all_models if not m["configured"]]
        if unconfigured:
            st.info(f"🔑 Add keys to unlock: {', '.join(m['name'] for m in unconfigured)}")

        comp_question = st.text_area(
            "Question",
            placeholder="e.g. What were the key risk factors disclosed in the latest report?",
            height=90,
            key="compare_question",
        )

        sel_models = st.multiselect(
            "Select models to compare",
            list(model_options_full.keys()),
            default=list(model_options_full.keys()),
            key="compare_models",
        )

        compare_btn = st.button("⚖️  Compare Models", key="btn_compare")

        if compare_btn:
            if not comp_question.strip():
                st.warning("Please enter a question.")
            elif not sel_models:
                st.warning("Select at least one model.")
            elif st.session_state.vector_store is None:
                st.error("No documents loaded. Upload a PDF in Document Management first.")
            else:
                run_deep_search = get_mod("run_deep_search")
                if run_deep_search is None:
                    st.error("Agent unavailable.")
                else:
                    results = {}
                    prog = st.progress(0, text="Starting…")
                    for idx, model_name in enumerate(sel_models):
                        prog.progress(idx / len(sel_models), text=f"Running {model_name}…")
                        try:
                            t0 = time.time()
                            res = run_deep_search(
                                question=comp_question.strip(),
                                model_name=model_options_full[model_name],
                                vector_store=st.session_state.vector_store,
                            )
                            res["_elapsed"] = time.time() - t0
                            res["_model_display"] = model_name
                            results[model_name] = res
                        except Exception as exc:
                            results[model_name] = {"error": str(exc)}
                        prog.progress((idx + 1) / len(sel_models), text=f"Done {model_name}")
                    prog.empty()
                    st.session_state.last_comparison_results = results
                    st.success(f"Comparison complete across {len(results)} model(s)!")

    comp_results = st.session_state.last_comparison_results
    if comp_results:
        st.divider()
        st.markdown("### Side-by-Side Answers")
        cols = st.columns(len(comp_results))
        for col, (model_name, res) in zip(cols, comp_results.items(), strict=False):
            with col:
                if "error" in res:
                    st.markdown(
                        f'<div class="answer-card" style="border-color:#f85149"><b style="color:#f85149">{model_name}</b><p style="color:#f85149">{res["error"]}</p></div>',
                        unsafe_allow_html=True,
                    )
                    continue
                conf = res.get("confidence_score", 0.0)
                st.markdown(
                    f"""
                <div class="answer-card">
                    <div style="font-weight:700;color:#f0c040;margin-bottom:0.5rem">{model_name}</div>
                    {confidence_badge(conf)}
                    <div style="margin-top:0.6rem;font-size:0.8rem;color:#8b949e">
                        Iterations: {res.get("iteration_count", 0)} &nbsp;|&nbsp;
                        Time: {res.get("_elapsed", 0):.1f}s &nbsp;|&nbsp;
                        Citations: {len(res.get("citations", []))}
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )
                answer = res.get("final_answer", "")
                st.markdown(answer[:800] + ("…" if len(answer) > 800 else ""))

        st.markdown("### Summary")
        detect_hall = get_mod("detect_hallucinations")
        rows = []
        for model_name, res in comp_results.items():
            if "error" in res:
                rows.append(
                    {
                        "Model": model_name,
                        "Confidence": "—",
                        "Iterations": "—",
                        "Time (s)": "—",
                        "Citations": "—",
                        "Hallucination Risk": "ERROR",
                    }
                )
                continue
            hall_risk = "—"
            if detect_hall:
                try:
                    h = detect_hall(res.get("final_answer", ""), res.get("retrieved_documents", []))
                    hall_risk = f"{h['hallucination_rate']:.0%}"
                except Exception:
                    pass
            rows.append(
                {
                    "Model": model_name,
                    "Confidence": f"{res.get('confidence_score', 0):.1%}",
                    "Iterations": res.get("iteration_count", 0),
                    "Time (s)": f"{res.get('_elapsed', 0):.1f}",
                    "Citations": len(res.get("citations", [])),
                    "Hallucination Risk": hall_risk,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        valid = {k: v for k, v in comp_results.items() if "error" not in v}
        if len(valid) > 1:
            fig = go.Figure(
                go.Bar(
                    x=list(valid.keys()),
                    y=[v.get("confidence_score", 0) for v in valid.values()],
                    marker_color=[
                        "#f0c040"
                        if v.get("confidence_score", 0) >= 0.8
                        else "#58a6ff"
                        if v.get("confidence_score", 0) >= 0.5
                        else "#f85149"
                        for v in valid.values()
                    ],
                    text=[f"{v.get('confidence_score', 0):.0%}" for v in valid.values()],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Confidence Score by Model",
                yaxis=dict(range=[0, 1.1], tickformat=".0%"),
                plot_bgcolor="#1c2128",
                paper_bgcolor="#1c2128",
                font=dict(color="#c9d1d9"),
                showlegend=False,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DOCUMENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("## 📁 Document Management")
    st.markdown(
        "Upload and index PDFs — financial reports, annual filings, earnings transcripts, or any document corpus."
    )
    st.divider()

    FinancialPDFParser = get_mod("FinancialPDFParser")
    FinancialChunker = get_mod("FinancialChunker")
    FinancialVectorStore = get_mod("FinancialVectorStore")
    modules_ready = all([FinancialPDFParser, FinancialChunker, FinancialVectorStore])

    if not modules_ready:
        missing = [
            n
            for n in ["FinancialPDFParser", "FinancialChunker", "FinancialVectorStore"]
            if not get_mod(n)
        ]
        st.error(f"Ingestion modules unavailable: {', '.join(missing)}")
        for m in missing:
            if m in MODULE_ERRORS:
                st.code(MODULE_ERRORS[m])

    docs = st.session_state.ingested_docs
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Documents", len(docs))
    s2.metric("Total Chunks", f"{sum(d.get('chunks', 0) for d in docs):,}")
    s3.metric("Storage", f"{sum(d.get('size_kb', 0) for d in docs) / 1024:.1f} MB")
    s4.metric("Vector Store", "✅ Ready" if st.session_state.vector_store else "⬜ Empty")

    st.divider()

    if docs:
        st.markdown("### Indexed Documents")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Document": d["name"],
                        "Chunks": d.get("chunks", 0),
                        "Size (KB)": d.get("size_kb", 0),
                        "Years": ", ".join(str(y) for y in d.get("years", [])) or "N/A",
                        "Ingested": d.get("ingested_at", "N/A"),
                    }
                    for d in docs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("🗑️  Clear All Documents", key="btn_clear"):
            st.session_state.confirm_clear = True
        if st.session_state.confirm_clear:
            st.warning("This will remove all indexed documents. Are you sure?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, clear all", key="btn_clear_yes"):
                    st.session_state.vector_store = None
                    st.session_state.ingested_docs = []
                    st.session_state.confirm_clear = False
                    st.rerun()
            with c2:
                if st.button("Cancel", key="btn_clear_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()
    else:
        st.info("No documents loaded yet. Upload PDFs below.")

    st.divider()
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.markdown(f"  • `{f.name}` — {len(f.getvalue()) / 1024:.0f} KB")

    ingest_btn = st.button(
        "📥  Ingest Documents", key="btn_ingest", disabled=not (uploaded_files and modules_ready)
    )

    if ingest_btn and uploaded_files and modules_ready:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "temp_uploads")
        os.makedirs(data_dir, exist_ok=True)

        parser = FinancialPDFParser()
        chunker = FinancialChunker()

        if st.session_state.vector_store is None:
            try:
                st.session_state.vector_store = FinancialVectorStore()
            except Exception as e:
                st.error(f"Failed to initialise vector store: {e}")
                st.stop()

        vs = st.session_state.vector_store
        progress_bar = st.progress(0, text="Preparing…")
        status_msg = st.empty()
        newly_ingested = []

        for i, f in enumerate(uploaded_files):
            progress_bar.progress(i / len(uploaded_files), text=f"Processing {f.name}…")
            tmp = os.path.join(data_dir, f.name)
            try:
                with open(tmp, "wb") as fh:
                    fh.write(f.getvalue())
                status_msg.markdown(f"📄 Parsing `{f.name}`…")
                parsed = parser.parse_file(tmp)
                status_msg.markdown(f"✂️ Chunking `{f.name}`…")
                chunks = chunker.chunk_documents(parsed)
                status_msg.markdown(f"🔍 Indexing `{f.name}`…")
                vs.ingest(chunks)
                years = sorted(
                    set(
                        c.get("metadata", {}).get("year", "")
                        for c in chunks
                        if c.get("metadata", {}).get("year")
                    )
                )
                newly_ingested.append(
                    {
                        "name": f.name,
                        "chunks": len(chunks),
                        "years": years,
                        "size_kb": round(len(f.getvalue()) / 1024, 1),
                        "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                )
            except Exception as exc:
                st.error(f"Failed to ingest `{f.name}`: {exc}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        progress_bar.progress(1.0, text="Done!")
        status_msg.empty()
        st.session_state.ingested_docs.extend(newly_ingested)
        if newly_ingested:
            st.success(
                f"✅ Ingested {len(newly_ingested)} document(s) — {sum(d['chunks'] for d in newly_ingested):,} chunks indexed."
            )
            st.rerun()

    with st.expander("ℹ️ Tips", expanded=False):
        st.markdown("""
        - Use text-based PDFs (not scanned images) for best extraction quality.
        - Annual reports, 10-Ks, earnings transcripts, and project docs all work well.
        - Upload multiple years for time-series and trend questions.
        - Financial tables are preserved and chunked with their surrounding context.
        """)
