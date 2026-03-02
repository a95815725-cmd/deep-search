"""
Section-aware, token-accurate chunker for financial documents.

Design goals
------------
* Tables are never split — each table is emitted as its own chunk.
* Regular prose is split on sentence boundaries first, then grouped into
  token-bounded chunks with configurable overlap.
* Every chunk carries rich metadata so the retrieval layer can filter by
  document, section, year, etc.
* Token counts use tiktoken (cl100k_base) so they match the OpenAI
  embedding model used downstream.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)

# Regex used to split prose into sentences.
# Uses a simple lookbehind for sentence-ending punctuation, then filters out
# common abbreviations in the splitting function below.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")

# Abbreviations that should NOT trigger a sentence split
_ABBREVS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "approx",
    "inc",
    "ltd",
    "corp",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
}


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, avoiding splits after common abbreviations."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    sentences: List[str] = []
    pending = ""
    for part in parts:
        candidate = (pending + " " + part).strip() if pending else part
        # Check if the split happened after an abbreviation
        last_word = (
            candidate.rstrip().rstrip(".").split()[-1].lower().rstrip(".")
            if candidate.strip()
            else ""
        )
        if last_word in _ABBREVS:
            pending = candidate
        else:
            sentences.append(candidate)
            pending = ""
    if pending:
        sentences.append(pending)
    return sentences if sentences else [text]


# Patterns used to extract a fiscal year from text or filenames.
_YEAR_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bfiscal\s+year\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bfy\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bfy\s*(\d{2})\b", re.IGNORECASE),  # FY24 → "24"
    re.compile(r"\byear\s+ended\s+\w+\s+\d{1,2},?\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bfor\s+the\s+year\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\b"),  # plain 4-digit year
    re.compile(r"[-_](20\d{2})[-_.]"),  # filename: report_2023.pdf
    re.compile(r"(20\d{2})"),  # last resort
]


class FinancialChunker:
    """
    Converts a list of page-level dicts (as produced by FinancialPDFParser)
    into a flat list of chunk dicts ready for vector-store ingestion.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._logger = logging.getLogger(self.__class__.__name__)

        # tiktoken encoder — cl100k_base is used by text-embedding-3-small
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "tiktoken unavailable, falling back to whitespace token count: %s", exc
            )
            self._enc = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk all pages from one or more documents.

        Parameters
        ----------
        parsed_pages:
            Output of ``FinancialPDFParser.parse_file`` or
            ``FinancialPDFParser.parse_directory``.

        Returns
        -------
        List of chunk dicts, each with keys:
            text, metadata (doc_name, page_num, section, chunk_id,
                            has_table, year)
        """
        all_chunks: List[Dict[str, Any]] = []
        for page_dict in parsed_pages:
            try:
                page_chunks = self.chunk_page(page_dict)
                all_chunks.extend(page_chunks)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Error chunking page %s/%s: %s",
                    page_dict.get("doc_name", "?"),
                    page_dict.get("page_num", "?"),
                    exc,
                )
        self._logger.info("Produced %d chunks from %d pages", len(all_chunks), len(parsed_pages))
        return all_chunks

    def chunk_page(self, page_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single page dict.

        Strategy
        --------
        1. Emit each table as its own chunk (never split a table).
        2. Strip table text from the page prose so it is not duplicated.
        3. Split remaining prose into sentences, then group into
           token-bounded chunks with overlap.
        4. Attach metadata to every chunk.
        """
        doc_name: str = page_dict.get("doc_name", "unknown")
        page_num: int = page_dict.get("page_num", 0)
        section: str = page_dict.get("section", "unknown")
        full_text: str = page_dict.get("text", "")
        tables: List[Dict[str, Any]] = page_dict.get("tables", [])

        year: Optional[str] = self.extract_year(full_text, doc_name)
        chunks: List[Dict[str, Any]] = []

        # ---- 1. Table chunks ------------------------------------------------
        prose_text = full_text
        for table in tables:
            table_text: str = table.get("raw_text", "")
            if not table_text.strip():
                continue

            chunk_id = self._make_chunk_id(doc_name, page_num, "table", table_text)
            chunks.append(
                {
                    "text": table_text,
                    "metadata": {
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "section": section,
                        "chunk_id": chunk_id,
                        "has_table": True,
                        "year": year,
                    },
                }
            )
            # Remove table text from prose to avoid duplication
            prose_text = prose_text.replace(table_text, "")

        # ---- 2. Prose chunks ------------------------------------------------
        prose_text = prose_text.strip()
        if prose_text:
            prose_chunks = self._chunk_prose(prose_text)
            for idx, prose_chunk_text in enumerate(prose_chunks):
                if not prose_chunk_text.strip():
                    continue
                chunk_id = self._make_chunk_id(doc_name, page_num, f"prose_{idx}", prose_chunk_text)
                chunks.append(
                    {
                        "text": prose_chunk_text,
                        "metadata": {
                            "doc_name": doc_name,
                            "page_num": page_num,
                            "section": section,
                            "chunk_id": chunk_id,
                            "has_table": False,
                            "year": year,
                        },
                    }
                )

        return chunks

    def extract_year(self, text: str, doc_name: str) -> Optional[str]:
        """
        Try to extract a fiscal/calendar year from the page text, then from
        the document filename.  Returns a 4-digit year string or None.
        """
        # Try text first
        for pattern in _YEAR_PATTERNS:
            match = pattern.search(text or "")
            if match:
                raw = match.group(1)
                year = self._normalise_year(raw)
                if year:
                    return year

        # Try filename
        for pattern in _YEAR_PATTERNS:
            match = pattern.search(doc_name or "")
            if match:
                raw = match.group(1)
                year = self._normalise_year(raw)
                if year:
                    return year

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in `text`."""
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Fallback: approximate via whitespace split
        return len(text.split())

    def _chunk_prose(self, text: str) -> List[str]:
        """
        Split prose text into chunks that respect `chunk_size` (in tokens)
        with `chunk_overlap` token overlap between consecutive chunks.

        Algorithm
        ---------
        1. Split text into sentences.
        2. Greedily accumulate sentences until the next sentence would
           exceed `chunk_size`.
        3. When flushing a chunk, carry forward the last N tokens' worth of
           sentences as overlap for the next chunk.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens: int = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If a single sentence exceeds chunk_size, hard-split it
            if sentence_tokens > self.chunk_size:
                # Flush what we have first
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences, current_tokens = self._get_overlap_sentences(
                        current_sentences
                    )
                # Hard-split the oversized sentence by words
                for sub in self._hard_split(sentence):
                    chunks.append(sub)
                continue

            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences, current_tokens = self._get_overlap_sentences(current_sentences)

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return [c.strip() for c in chunks if c.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, respecting common abbreviations."""
        sentences = _split_sentences(text)
        # Further split on newlines that indicate paragraph/section breaks
        result: List[str] = []
        for sentence in sentences:
            parts = [p.strip() for p in sentence.split("\n\n") if p.strip()]
            result.extend(parts)
        return result

    def _get_overlap_sentences(self, sentences: List[str]) -> tuple[List[str], int]:
        """
        From the tail of `sentences`, pick as many sentences as will fit
        within `chunk_overlap` tokens.  Returns (overlap_sentences, token_count).
        """
        overlap_sentences: List[str] = []
        overlap_tokens = 0
        for sentence in reversed(sentences):
            t = self._count_tokens(sentence)
            if overlap_tokens + t > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += t
        return overlap_sentences, overlap_tokens

    def _hard_split(self, text: str) -> List[str]:
        """
        Word-level split for text that exceeds chunk_size in a single
        sentence (rare in financial documents but can occur in data tables
        that weren't caught by the table extractor).
        """
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for word in words:
            wt = self._count_tokens(word)
            if current_tokens + wt > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Apply word-level overlap
                overlap: List[str] = []
                ot = 0
                for w in reversed(current):
                    wtt = self._count_tokens(w)
                    if ot + wtt > self.chunk_overlap:
                        break
                    overlap.insert(0, w)
                    ot += wtt
                current = overlap
                current_tokens = ot
            current.append(word)
            current_tokens += wt

        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _make_chunk_id(doc_name: str, page_num: int, kind: str, text: str) -> str:
        """Stable, deterministic chunk identifier based on content."""
        payload = f"{doc_name}::{page_num}::{kind}::{text[:200]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _normalise_year(raw: str) -> Optional[str]:
        """
        Convert a raw year string to a 4-digit year string.
        Handles 2-digit abbreviations (e.g. "24" → "2024").
        Rejects years outside the plausible range 1900–2099.
        """
        if len(raw) == 2:
            # Assume 2000s for values ≤ 50, else 1900s
            year_int = int(raw)
            full = 2000 + year_int if year_int <= 50 else 1900 + year_int
            return str(full)
        if len(raw) == 4:
            year_int = int(raw)
            if 1900 <= year_int <= 2099:
                return raw
        return None
