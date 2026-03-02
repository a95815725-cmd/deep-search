"""
PDF parser specialised for financial documents (annual reports, earnings
call transcripts, 10-K / 10-Q filings, etc.).

Primary engine: pdfplumber
Fallback engine: pypdf (for scanned / image-heavy PDFs that pdfplumber cannot
                         extract text from)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section-detection keyword map
# Keys are canonical section names; values are lists of keyword phrases that,
# when found in the page text (case-insensitive), indicate that section.
# The order matters: more-specific patterns should come first.
# ---------------------------------------------------------------------------
_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "balance_sheet": [
        "balance sheet",
        "statement of financial position",
        "financial position",
        "total assets",
        "total liabilities",
        "shareholders' equity",
        "stockholders' equity",
        "equity attributable",
    ],
    "income_statement": [
        "income statement",
        "statement of income",
        "statement of operations",
        "profit and loss",
        "profit & loss",
        "net income",
        "net profit",
        "net loss",
        "earnings per share",
        "total revenue",
        "gross profit",
        "operating income",
        "operating profit",
    ],
    "cash_flow": [
        "cash flow",
        "statement of cash flows",
        "cash and cash equivalents",
        "operating activities",
        "investing activities",
        "financing activities",
        "free cash flow",
        "capital expenditure",
        "capex",
    ],
    "risk_factors": [
        "risk factors",
        "principal risks",
        "key risks",
        "material risks",
        "risk management",
        "market risk",
        "credit risk",
        "liquidity risk",
        "operational risk",
    ],
    "management_discussion": [
        "management's discussion",
        "management discussion",
        "md&a",
        "management report",
        "ceo letter",
        "chairman's statement",
        "chief executive",
        "outlook",
        "strategy overview",
        "business review",
    ],
    "earnings_call": [
        "earnings call",
        "earnings conference",
        "conference call",
        "q&a session",
        "question and answer",
        "analyst question",
        "operator:",
        "moderator:",
    ],
    "notes": [
        "notes to the",
        "note 1",
        "note 2",
        "note 3",
        "accounting policies",
        "significant accounting",
        "critical accounting",
        "basis of preparation",
        "consolidation",
        "ifrs",
        "gaap",
    ],
}


class FinancialPDFParser:
    """
    Parses financial PDF documents into structured page-level dictionaries
    that downstream chunkers and vector stores can consume.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a single PDF file.

        Returns a list of page-level dicts:
        {
            "text":     str,           # full extracted text of the page
            "tables":   List[Dict],    # tables extracted from the page
            "page_num": int,           # 1-based page number
            "section":  str,           # detected financial section label
            "doc_name": str,           # stem of the filename (no extension)
        }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        doc_name = path.stem
        pages: List[Dict[str, Any]] = []

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    try:
                        page_dict = self._parse_page(page, page_number, doc_name)
                        pages.append(page_dict)
                    except Exception as exc:  # noqa: BLE001
                        self._logger.warning(
                            "Error parsing page %d of %s: %s",
                            page_number,
                            file_path,
                            exc,
                        )
                        # Append a minimal stub so page numbering stays intact
                        pages.append(
                            {
                                "text": "",
                                "tables": [],
                                "page_num": page_number,
                                "section": "unknown",
                                "doc_name": doc_name,
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to open %s with pdfplumber: %s", file_path, exc)
            # Attempt graceful fallback using pypdf
            pages = self._fallback_pypdf(path, doc_name)

        self._logger.info("Parsed %d pages from '%s'", len(pages), path.name)
        return pages

    def parse_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Parse every PDF found (recursively) under `dir_path`.
        Returns a flat list of page dicts across all documents.
        """
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_pages: List[Dict[str, Any]] = []
        pdf_files = sorted(directory.rglob("*.pdf"))

        if not pdf_files:
            self._logger.warning("No PDF files found under %s", dir_path)
            return all_pages

        for pdf_path in pdf_files:
            try:
                pages = self.parse_file(str(pdf_path))
                all_pages.extend(pages)
                self._logger.info("Ingested %d pages from %s", len(pages), pdf_path.name)
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Skipping %s due to error: %s", pdf_path.name, exc)

        return all_pages

    def extract_tables(self, page: Any) -> List[Dict[str, Any]]:
        """
        Extract tables from a pdfplumber page object.

        Returns a list of table dicts:
        {
            "headers":  List[str],   # first row treated as header
            "rows":     List[List[str]],
            "raw_text": str,         # pipe-separated human-readable form
        }
        """
        tables: List[Dict[str, Any]] = []
        try:
            raw_tables = page.extract_tables()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Table extraction failed: %s", exc)
            return tables

        if not raw_tables:
            return tables

        for raw_table in raw_tables:
            if not raw_table:
                continue

            # Clean cells: replace None with empty string, strip whitespace
            cleaned: List[List[str]] = []
            for row in raw_table:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                cleaned.append(cleaned_row)

            if not cleaned:
                continue

            headers = cleaned[0]
            rows = cleaned[1:] if len(cleaned) > 1 else []
            raw_text = self._table_to_string(headers, rows)

            tables.append(
                {
                    "headers": headers,
                    "rows": rows,
                    "raw_text": raw_text,
                }
            )

        return tables

    def detect_section(self, text: str) -> str:
        """
        Detect which financial section a block of text belongs to.

        Detection strategy:
        1. Lower-case the text and search for each section's keywords.
        2. Count matches per section; return the section with the most hits.
        3. Fall back to "unknown" if no keywords match.
        """
        if not text or not text.strip():
            return "unknown"

        text_lower = text.lower()
        scores: Dict[str, int] = {section: 0 for section in _SECTION_KEYWORDS}

        for section, keywords in _SECTION_KEYWORDS.items():
            for kw in keywords:
                # Use word-boundary-like matching via a simple count
                if kw in text_lower:
                    scores[section] += 1

        best_section = max(scores, key=lambda s: scores[s])
        if scores[best_section] == 0:
            return "unknown"
        return best_section

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_page(self, page: Any, page_number: int, doc_name: str) -> Dict[str, Any]:
        """Extract all content from a single pdfplumber page."""
        # Extract tables first (before text, so we can note their presence)
        tables = self.extract_tables(page)

        # Extract text.  `layout=True` preserves column order better for
        # multi-column financial statements.
        text: str = page.extract_text(layout=False) or ""

        # Append table raw_text so that the section detector can see it,
        # and so the text is searchable even if not in the page's text layer.
        table_texts = [t["raw_text"] for t in tables if t.get("raw_text")]
        if table_texts:
            combined_table_text = "\n\n".join(table_texts)
            text = text + "\n\n" + combined_table_text if text else combined_table_text

        text = text.strip()
        section = self.detect_section(text)

        return {
            "text": text,
            "tables": tables,
            "page_num": page_number,
            "section": section,
            "doc_name": doc_name,
        }

    def _table_to_string(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Format a table as a pipe-delimited string, e.g.:

            | Revenue | 2023 | 2022 |
            |---------|------|------|
            | Total   | 100  |  90  |
        """
        if not headers and not rows:
            return ""

        # Determine column widths
        all_rows = [headers] + rows if headers else rows
        col_count = max(len(r) for r in all_rows)

        # Pad all rows to the same column count
        padded = [r + [""] * (col_count - len(r)) for r in all_rows]

        col_widths = [
            max(len(str(padded[r][c])) for r in range(len(padded))) for c in range(col_count)
        ]

        def format_row(row: List[str]) -> str:
            cells = [str(row[c]).ljust(col_widths[c]) for c in range(col_count)]
            return "| " + " | ".join(cells) + " |"

        lines: List[str] = []
        if headers:
            lines.append(format_row(padded[0]))
            separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
            lines.append(separator)
            for row in padded[1:]:
                lines.append(format_row(row))
        else:
            for row in padded:
                lines.append(format_row(row))

        return "\n".join(lines)

    def _fallback_pypdf(self, path: Path, doc_name: str) -> List[Dict[str, Any]]:
        """
        Attempt text extraction using pypdf when pdfplumber fails entirely.
        Returns page dicts without table data.
        """
        pages: List[Dict[str, Any]] = []
        try:
            from pypdf import PdfReader  # lazy import to keep startup fast

            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    section = self.detect_section(text)
                    pages.append(
                        {
                            "text": text,
                            "tables": [],
                            "page_num": page_number,
                            "section": section,
                            "doc_name": doc_name,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning("pypdf fallback: error on page %d: %s", page_number, exc)
                    pages.append(
                        {
                            "text": "",
                            "tables": [],
                            "page_num": page_number,
                            "section": "unknown",
                            "doc_name": doc_name,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("pypdf fallback also failed for %s: %s", path.name, exc)
        return pages
