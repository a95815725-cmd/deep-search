"""
Deep Search MCP Server
======================
Exposes the financial deep search agent as MCP tools so any MCP-compatible
agent (Claude Desktop, PR Review Agent, Backlog Assistant, etc.) can call
deep research as a capability.

Tools exposed:
  - deep_search          : Full iterative agent loop (planner→search→reflect→synthesize)
  - search_docs          : Direct vector search (fast, no agent loop)
  - ingest_document      : Add a PDF to the knowledge base
  - list_knowledge_base  : What documents + years are indexed

Resources exposed:
  - docs://corpus        : Summary of all ingested documents

Usage:
  python mcp_server.py                        # stdio (Claude Desktop, MCP clients)
  python mcp_server.py --transport sse        # SSE (HTTP-based agents)

"""

import asyncio
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("deep-search-mcp")

# ---------------------------------------------------------------------------
# Lazy singletons — only initialised on first tool call to keep startup fast
# ---------------------------------------------------------------------------
_vector_store = None
_graph_ready = False


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from src.ingestion.vector_store import FinancialVectorStore

        _vector_store = FinancialVectorStore()
        logger.info("FinancialVectorStore initialised (%d chunks)", _vector_store.count())
    return _vector_store


def _run_deep_search(question: str, model_name: str = "gpt-4o") -> dict:
    from src.agent.graph import run_deep_search

    vs = _get_vector_store()
    return run_deep_search(
        question=question,
        model_name=model_name,
        vector_store=vs,
    )


# ---------------------------------------------------------------------------
# MCP Server definition
# ---------------------------------------------------------------------------

server = Server("deep-search-agent")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="deep_search",
            description=(
                "Run an iterative deep search over the ingested document corpus. "
                "The agent decomposes the question into sub-queries, searches iteratively, "
                "reflects on gaps, and synthesises a grounded answer with citations. "
                "Use this for complex, multi-part questions that need comprehensive research. "
                "Returns: answer, citations, confidence_score, iterations_used, reasoning_trace."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The research question to answer from the document corpus.",
                    },
                    "model": {
                        "type": "string",
                        "description": "LLM to use: 'gpt-4o' (default), 'gpt-4o-mini', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash'",
                        "default": "gpt-4o",
                    },
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="search_docs",
            description=(
                "Fast direct vector search over the document corpus — no agent loop. "
                "Returns the top-K most relevant chunks with metadata. "
                "Use this for quick lookups when you need raw passages, not a synthesised answer. "
                "Set summary_only=true to return just doc/page/score metadata without chunk content "
                "(useful for corpus exploration before deciding whether to call deep_search)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant document chunks.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20).",
                        "default": 5,
                    },
                    "year": {
                        "type": "string",
                        "description": "Optional: filter results to a specific fiscal year (e.g. '2023').",
                    },
                    "summary_only": {
                        "type": "boolean",
                        "description": (
                            "If true, return only doc name, page, section, and relevance score — "
                            "no chunk content. Reduces tokens when you only need to know which "
                            "documents are relevant before calling deep_search."
                        ),
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ingest_document",
            description=(
                "Add a PDF document to the deep search knowledge base. "
                "The document will be parsed, chunked with section-awareness, and indexed. "
                "After ingestion it becomes immediately searchable by all other tools."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to a PDF file to ingest.",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="list_knowledge_base",
            description=(
                "List all documents currently indexed in the knowledge base. "
                "Returns document names, chunk counts, and available fiscal years. "
                "Call this first to understand what context is available before searching."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    # -----------------------------------------------------------------------
    if name == "deep_search":
        question = arguments.get("question", "").strip()
        model = arguments.get("model", "gpt-4o")

        if not question:
            return [types.TextContent(type="text", text="Error: question is required.")]

        logger.info("deep_search called | model=%s | question=%.80s", model, question)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _run_deep_search(question, model)
            )
        except Exception as exc:
            logger.error("deep_search failed: %s", exc)
            return [types.TextContent(type="text", text=f"Deep search failed: {exc}")]

        # Format output for the calling agent
        answer = result.get("answer", "No answer generated.")
        confidence = result.get("confidence_score", 0.0)
        iterations = result.get("iteration_count", 0)
        citations = result.get("citations", [])
        gaps = result.get("gaps_identified", [])

        output_lines = [
            "## Deep Search Result",
            "",
            f"**Confidence:** {confidence:.0%}  |  **Iterations:** {iterations}  |  **Model:** {model}",
            "",
            "### Answer",
            answer,
        ]

        if citations:
            output_lines += ["", "### Citations"]
            for i, c in enumerate(citations[:8], 1):
                output_lines.append(
                    f"{i}. **{c.get('doc_name', '?')}** p.{c.get('page_num', '?')} "
                    f"[{c.get('section', '?')}] — {c.get('text_excerpt', '')[:120]}"
                )

        if gaps:
            output_lines += ["", "### Unresolved Gaps"]
            for g in gaps:
                output_lines.append(f"- {g}")

        return [types.TextContent(type="text", text="\n".join(output_lines))]

    # -----------------------------------------------------------------------
    elif name == "search_docs":
        query = arguments.get("query", "").strip()
        top_k = min(int(arguments.get("top_k", 5)), 20)
        year = arguments.get("year", "").strip() or None
        summary_only = bool(arguments.get("summary_only", False))

        if not query:
            return [types.TextContent(type="text", text="Error: query is required.")]

        logger.info(
            "search_docs called | query=%.60s | top_k=%d | year=%s | summary_only=%s",
            query,
            top_k,
            year,
            summary_only,
        )

        try:
            vs = _get_vector_store()
            if year:
                results = vs.search_with_year(query=query, year=year, top_k=top_k)
            else:
                results = vs.search(query=query, top_k=top_k)
        except Exception as exc:
            return [types.TextContent(type="text", text=f"Search failed: {exc}")]

        if not results:
            return [types.TextContent(type="text", text="No relevant documents found.")]

        lines = [f"## Search Results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            score = r.get("score", 0.0)
            header = (
                f"{i}. {meta.get('doc_name', '?')}  p.{meta.get('page_num', '?')}"
                f"  [{meta.get('section', '?')}]  score={score:.3f}"
                + (f"  year={meta.get('year')}" if meta.get("year") else "")
            )
            lines.append(header)
            if not summary_only:
                lines.append(r.get("text", "")[:600])
                lines.append("")

        if summary_only:
            lines.append(
                f"\n({len(results)} results — use summary_only=false or deep_search for full content)"
            )

        return [types.TextContent(type="text", text="\n".join(lines))]

    # -----------------------------------------------------------------------
    elif name == "ingest_document":
        file_path = arguments.get("file_path", "").strip()

        if not file_path:
            return [types.TextContent(type="text", text="Error: file_path is required.")]
        if not Path(file_path).exists():
            return [types.TextContent(type="text", text=f"Error: file not found: {file_path}")]
        if not file_path.lower().endswith(".pdf"):
            return [types.TextContent(type="text", text="Error: only PDF files are supported.")]

        logger.info("ingest_document called | path=%s", file_path)

        try:
            from src.ingestion.chunker import FinancialChunker
            from src.ingestion.pdf_parser import FinancialPDFParser

            parser = FinancialPDFParser()
            chunker = FinancialChunker()
            vs = _get_vector_store()

            parsed = parser.parse_file(file_path)
            chunks = chunker.chunk_documents(parsed)
            stored = vs.ingest(chunks)

            years = sorted(
                set(
                    c.get("metadata", {}).get("year", "")
                    for c in chunks
                    if c.get("metadata", {}).get("year")
                )
            )

            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"✅ Ingested **{Path(file_path).name}**\n"
                        f"- Pages parsed: {len(parsed)}\n"
                        f"- Chunks stored: {stored}\n"
                        f"- Years detected: {', '.join(years) if years else 'none detected'}\n"
                        f"- Total corpus size: {vs.count()} chunks"
                    ),
                )
            ]
        except Exception as exc:
            logger.error("ingest_document failed: %s", exc)
            return [types.TextContent(type="text", text=f"Ingestion failed: {exc}")]

    # -----------------------------------------------------------------------
    elif name == "list_knowledge_base":
        logger.info("list_knowledge_base called")

        try:
            vs = _get_vector_store()
            docs = vs.get_available_docs()
            years = vs.get_available_years()
            total = vs.count()
        except Exception as exc:
            return [types.TextContent(type="text", text=f"Failed to list knowledge base: {exc}")]

        if not docs:
            return [
                types.TextContent(
                    type="text",
                    text="Knowledge base is empty. Use `ingest_document` to add documents.",
                )
            ]

        lines = [
            "## Knowledge Base Summary",
            "",
            f"**Total chunks indexed:** {total}",
            f"**Available years:** {', '.join(sorted(years)) if years else 'none detected'}",
            "",
            f"### Documents ({len(docs)})",
        ]
        for doc in sorted(docs):
            lines.append(f"- {doc}")

        return [types.TextContent(type="text", text="\n".join(lines))]

    # -----------------------------------------------------------------------
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Resources — expose corpus summary as a readable resource
# ---------------------------------------------------------------------------


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="docs://corpus",
            name="Document Corpus",
            description="Summary of all documents ingested into the deep search knowledge base.",
            mimeType="text/plain",
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "docs://corpus":
        try:
            vs = _get_vector_store()
            docs = vs.get_available_docs()
            years = vs.get_available_years()
            total = vs.count()
            lines = [
                "Deep Search Knowledge Base",
                f"Total chunks: {total}",
                f"Years: {', '.join(sorted(years)) if years else 'none'}",
                "Documents:",
            ] + [f"  - {d}" for d in sorted(docs)]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error reading corpus: {exc}"
    raise ValueError(f"Unknown resource: {uri}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main():
    logger.info("Starting Deep Search MCP Server (stdio)…")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="deep-search-agent",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
