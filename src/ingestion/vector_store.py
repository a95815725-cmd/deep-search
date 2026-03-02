"""
ChromaDB vector store wrapper optimised for financial document search.

Features
--------
* Persistent on-disk storage via ChromaDB's built-in persistence.
* Embeddings via OpenAI text-embedding-3-small (langchain-openai wrapper).
* Metadata filtering: by arbitrary field, or by year.
* Helper queries: list all ingested document names, list all available years.
* Batch ingestion with progress logging.
* Thread-safe: uses a single ChromaDB PersistentClient.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Maximum number of chunks sent to ChromaDB in a single upsert call.
# ChromaDB can handle larger batches, but keeping this bounded avoids
# memory spikes when ingesting very large documents.
_INGEST_BATCH_SIZE = 500


class FinancialVectorStore:
    """
    Thin, opinionated wrapper around ChromaDB for financial document search.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB stores its on-disk data.
        Defaults to ``./data/chroma_db`` (relative to cwd at construction
        time).  The directory is created if it does not exist.
    collection_name:
        Name of the ChromaDB collection.  A new collection is created
        automatically on first use.
    openai_api_key:
        OpenAI API key used for embedding.  Falls back to the
        ``OPENAI_API_KEY`` environment variable if not supplied.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "financial_docs",
        openai_api_key: Optional[str] = None,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

        # ---- Resolve & create persist directory ----------------------------
        if persist_dir is None:
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        self._persist_dir = str(Path(persist_dir).resolve())
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._collection_name = collection_name

        # ---- Embeddings ----------------------------------------------------
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,  # type: ignore[arg-type]
        )

        # ---- ChromaDB client & collection ----------------------------------
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection()
        self._logger.info(
            "FinancialVectorStore ready — collection=%r, persist_dir=%r, existing chunks=%d",
            self._collection_name,
            self._persist_dir,
            self.count(),
        )

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    def ingest(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and store a list of chunk dicts.

        Each chunk dict must have:
            ``text``     – the string to embed
            ``metadata`` – a dict of scalar metadata values

        Returns the number of chunks successfully stored.
        """
        if not chunks:
            return 0

        stored = 0
        total = len(chunks)
        self._logger.info("Ingesting %d chunks …", total)

        for batch_start in range(0, total, _INGEST_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _INGEST_BATCH_SIZE]

            texts = [c["text"] for c in batch if c.get("text", "").strip()]
            if not texts:
                continue

            # Filter out chunks with empty text (they were already excluded above,
            # but keep the index aligned)
            valid_batch = [c for c in batch if c.get("text", "").strip()]

            try:
                embeddings = self._embeddings.embed_documents(texts)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Embedding failed for batch %d–%d: %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                continue

            ids: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            documents: List[str] = []

            for chunk, _embedding in zip(valid_batch, embeddings, strict=False):
                chunk_id = chunk.get("metadata", {}).get("chunk_id") or str(uuid.uuid4())
                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append(self._sanitise_metadata(chunk.get("metadata", {})))

            try:
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                stored += len(ids)
                self._logger.debug(
                    "Upserted batch %d–%d (%d chunks)",
                    batch_start,
                    batch_start + len(batch),
                    len(ids),
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "ChromaDB upsert failed for batch %d–%d: %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )

        self._logger.info("Ingestion complete — %d / %d chunks stored", stored, total)
        return stored

    def clear(self) -> None:
        """Delete all documents in the collection and recreate it."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._get_or_create_collection()
        self._logger.info("Collection %r cleared.", self._collection_name)

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 8,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over all ingested chunks.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of results to return.
        filter_metadata:
            Optional ChromaDB ``where`` clause dict, e.g.
            ``{"section": "balance_sheet"}`` or
            ``{"$and": [{"year": "2023"}, {"has_table": True}]}``.

        Returns
        -------
        List of result dicts:
            ``text``     – chunk text
            ``metadata`` – chunk metadata
            ``score``    – cosine distance (lower = more similar; range 0–2)
        """
        if not query.strip():
            return []

        try:
            query_embedding = self._embeddings.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to embed query: %s", exc)
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(self.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        try:
            results = self._collection.query(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("ChromaDB query failed: %s", exc)
            return []

        return self._format_results(results)

    def search_with_year(self, query: str, year: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Convenience wrapper: search restricted to chunks from a specific year.
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter_metadata={"year": str(year)},
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_available_docs(self) -> List[str]:
        """Return a sorted, deduplicated list of ingested document names."""
        return self._get_unique_metadata_values("doc_name")

    def get_available_years(self) -> List[str]:
        """Return a sorted, deduplicated list of years present in the store."""
        years = self._get_unique_metadata_values("year")
        # Filter out None / empty strings
        return sorted(y for y in years if y)

    def count(self) -> int:
        """Total number of chunks currently stored in the collection."""
        try:
            return self._collection.count()
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create a new one (cosine distance)."""
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_unique_metadata_values(self, field: str) -> List[str]:
        """
        Retrieve all unique values of a metadata field across the collection.
        ChromaDB does not expose a ``distinct`` API, so we fetch all items
        and deduplicate in Python.  Fine for POC scale (< 100 k chunks).
        """
        try:
            result = self._collection.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            values = {str(m.get(field, "")) for m in metadatas if m and m.get(field) is not None}
            return sorted(values)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Could not retrieve unique values for field %r: %s", field, exc)
            return []

    @staticmethod
    def _sanitise_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB only accepts str, int, float, or bool metadata values.
        Convert anything else to a string.  None becomes the empty string
        so the key is preserved and can be filtered on.
        """
        sanitised: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                sanitised[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitised[key] = value
            else:
                sanitised[key] = str(value)
        return sanitised

    @staticmethod
    def _format_results(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transform a ChromaDB query response into a clean list of result dicts.
        """
        formatted: List[Dict[str, Any]] = []

        documents_batch = raw.get("documents") or [[]]
        metadatas_batch = raw.get("metadatas") or [[]]
        distances_batch = raw.get("distances") or [[]]

        documents = documents_batch[0] if documents_batch else []
        metadatas = metadatas_batch[0] if metadatas_batch else []
        distances = distances_batch[0] if distances_batch else []

        for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
            formatted.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "score": round(float(dist), 6),
                }
            )

        return formatted
