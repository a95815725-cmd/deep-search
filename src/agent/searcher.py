import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Set, Tuple

from src.agent.state import AgentState, SubQuery

logger = logging.getLogger(__name__)


def _content_hash(doc: Dict) -> str:
    """Stable hash of a document dict for deduplication."""
    content = doc.get("content", "")
    source = str(doc.get("metadata", {}).get("source", ""))
    page = str(doc.get("metadata", {}).get("page", ""))
    raw = f"{content[:500]}|{source}|{page}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _search_vector_store(vector_store: Any, query: str, k: int = 6) -> List[Dict]:
    """Call FinancialVectorStore.search() and normalise to the agent's doc format."""
    try:
        return [
            {
                "content": r["text"],
                "metadata": r.get("metadata", {}),
                "source": "vector_store",
                "score": r.get("score"),
            }
            for r in vector_store.search(query=query, top_k=k)
        ]
    except Exception as exc:
        logger.warning("Vector store search failed for query '%s': %s", query[:60], exc)
        return []


def _resolve_vector_store(configurable: Dict) -> Any:
    """
    Resolve the vector store from LangGraph configurable dict,
    or instantiate a default one.
    """
    vs = configurable.get("vector_store")
    if vs is not None:
        return vs

    # Fall back to creating one with default settings
    try:
        from src.ingestion.vector_store import FinancialVectorStore  # type: ignore

        vs = FinancialVectorStore()
        logger.info("Searcher: instantiated default FinancialVectorStore.")
        return vs
    except Exception as exc:
        logger.error("Could not create FinancialVectorStore: %s", exc)
        raise RuntimeError(
            "No vector_store provided in configurable and default instantiation failed."
        ) from exc


async def _search_one_query_async(
    sq: SubQuery,
    vector_store: Any,
) -> Tuple[SubQuery, List[Dict]]:
    """
    Run vector store search for a single sub-query via asyncio.to_thread.
    ChromaDB's search() is synchronous — to_thread offloads it to the thread pool
    so all sub-queries execute concurrently rather than one after another.

    Returns (sub_query, raw_results) — deduplication happens after all queries finish.
    """
    query = sq["query"]
    logger.info("Searcher: queuing sub-query: %.80s", query)
    docs = await asyncio.to_thread(_search_vector_store, vector_store, query, 6)
    logger.debug("  '%.40s': %d docs returned", query, len(docs))
    return sq, docs


async def _search_all_pending_async(
    pending_sqs: List[SubQuery],
    vector_store: Any,
) -> List[Tuple[SubQuery, List[Dict]]]:
    """
    Fire all pending sub-queries in parallel.
    asyncio.gather preserves input order in the returned results.
    """
    tasks = [_search_one_query_async(sq, vector_store) for sq in pending_sqs]
    return await asyncio.gather(*tasks)


def searcher_node(state: AgentState, config: dict) -> dict:
    """
    Executes all pending sub-queries against the vector store in parallel.

    All N sub-query searches fire simultaneously via asyncio.gather + asyncio.to_thread.
    Before: N serial ChromaDB calls.  After: all N calls run concurrently.

    New, deduplicated documents are appended to retrieved_documents via operator.add.
    """
    configurable = (config or {}).get("configurable", {})
    vector_store = _resolve_vector_store(configurable)

    # Build dedup set from documents already accumulated in state
    existing_hashes: Set[str] = {_content_hash(d) for d in state.get("retrieved_documents", [])}

    pending_sqs = [sq for sq in state["sub_queries"] if sq["status"] == "pending"]
    already_searched = [sq for sq in state["sub_queries"] if sq["status"] != "pending"]

    # ── Parallel search ───────────────────────────────────────────────────────
    query_results: List[Tuple[SubQuery, List[Dict]]] = asyncio.run(
        _search_all_pending_async(pending_sqs, vector_store)
    )

    # ── Serial deduplication (hash set mutation must not be concurrent) ───────
    all_new_docs: List[Dict] = []
    searched_sqs: List[SubQuery] = []

    for sq, raw_results in query_results:
        unique_results: List[Dict] = []
        for doc in raw_results:
            h = _content_hash(doc)
            if h not in existing_hashes:
                existing_hashes.add(h)
                unique_results.append(doc)
                all_new_docs.append(doc)
        searched_sqs.append(SubQuery(query=sq["query"], status="searched", results=unique_results))

    updated_sub_queries = already_searched + searched_sqs
    iteration_count = state.get("iteration_count", 0) + 1

    trace_entry = (
        f"Searcher [iter {iteration_count}]: {len(pending_sqs)} sub-queries in parallel, "
        f"{len(all_new_docs)} new unique docs"
    )
    logger.info(trace_entry)

    return {
        "sub_queries": updated_sub_queries,
        "retrieved_documents": all_new_docs,  # operator.add appends to existing list
        "iteration_count": iteration_count,
        "reasoning_trace": [trace_entry],
    }
