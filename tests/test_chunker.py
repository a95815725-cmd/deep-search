"""
Unit tests for src/ingestion/chunker.py

Tests cover:
  - extract_year: various text and filename formats
  - _normalise_year: 2-digit, 4-digit, out-of-range
  - chunk_page: metadata shape, table chunks separated, prose chunks produced
  - _make_chunk_id: determinism and uniqueness
"""

import pytest

from src.ingestion.chunker import FinancialChunker


@pytest.fixture
def chunker() -> FinancialChunker:
    return FinancialChunker(chunk_size=200, chunk_overlap=40)


# ---------------------------------------------------------------------------
# _normalise_year
# ---------------------------------------------------------------------------


class TestNormaliseYear:
    def test_4digit_valid(self):
        assert FinancialChunker._normalise_year("2023") == "2023"

    def test_4digit_edge_low(self):
        assert FinancialChunker._normalise_year("1900") == "1900"

    def test_4digit_edge_high(self):
        assert FinancialChunker._normalise_year("2099") == "2099"

    def test_4digit_out_of_range_returns_none(self):
        assert FinancialChunker._normalise_year("1899") is None
        assert FinancialChunker._normalise_year("2100") is None

    def test_2digit_low_maps_to_2000s(self):
        assert FinancialChunker._normalise_year("23") == "2023"
        assert FinancialChunker._normalise_year("00") == "2000"
        assert FinancialChunker._normalise_year("50") == "2050"

    def test_2digit_high_maps_to_1900s(self):
        assert FinancialChunker._normalise_year("51") == "1951"
        assert FinancialChunker._normalise_year("99") == "1999"


# ---------------------------------------------------------------------------
# extract_year
# ---------------------------------------------------------------------------


class TestExtractYear:
    def test_plain_year_in_text(self, chunker):
        assert chunker.extract_year("Annual Report 2023", "doc") == "2023"

    def test_fiscal_year_pattern(self, chunker):
        assert chunker.extract_year("For fiscal year 2022 the results were...", "doc") == "2022"

    def test_fy_abbreviation(self, chunker):
        assert chunker.extract_year("FY2024 results exceeded expectations.", "doc") == "2024"

    def test_2digit_fy_abbreviation(self, chunker):
        assert chunker.extract_year("FY24 annual report.", "doc") == "2024"

    def test_year_from_filename_fallback(self, chunker):
        # No year in text, but year in doc name
        assert chunker.extract_year("No year mentioned here.", "rabobank_annual_2021") == "2021"

    def test_year_in_filename_with_dashes(self, chunker):
        assert chunker.extract_year("", "report-2020-q4") == "2020"

    def test_no_year_returns_none(self, chunker):
        assert chunker.extract_year("No dates in this text at all.", "some_document") is None

    def test_empty_inputs_return_none(self, chunker):
        assert chunker.extract_year("", "") is None


# ---------------------------------------------------------------------------
# chunk_page
# ---------------------------------------------------------------------------


class TestChunkPage:
    def _make_page(self, text="Sample text. " * 20, tables=None, section="income_statement"):
        return {
            "text": text,
            "tables": tables or [],
            "page_num": 5,
            "section": section,
            "doc_name": "test_doc",
        }

    def test_chunks_have_required_metadata_keys(self, chunker):
        page = self._make_page()
        chunks = chunker.chunk_page(page)
        assert len(chunks) > 0
        for chunk in chunks:
            meta = chunk["metadata"]
            assert "doc_name" in meta
            assert "page_num" in meta
            assert "section" in meta
            assert "chunk_id" in meta
            assert "has_table" in meta

    def test_doc_name_and_page_propagated(self, chunker):
        page = self._make_page()
        chunks = chunker.chunk_page(page)
        for chunk in chunks:
            assert chunk["metadata"]["doc_name"] == "test_doc"
            assert chunk["metadata"]["page_num"] == 5
            assert chunk["metadata"]["section"] == "income_statement"

    def test_table_chunk_has_table_true(self, chunker):
        table = {"raw_text": "| Revenue | 100 |\n| Costs | 80 |", "headers": [], "rows": []}
        page = self._make_page(tables=[table])
        chunks = chunker.chunk_page(page)
        table_chunks = [c for c in chunks if c["metadata"]["has_table"]]
        assert len(table_chunks) == 1
        assert "Revenue" in table_chunks[0]["text"]

    def test_prose_chunks_have_table_false(self, chunker):
        page = self._make_page(text="Net income was strong. " * 10)
        chunks = chunker.chunk_page(page)
        prose_chunks = [c for c in chunks if not c["metadata"]["has_table"]]
        assert len(prose_chunks) > 0
        for c in prose_chunks:
            assert c["metadata"]["has_table"] is False

    def test_empty_page_returns_no_chunks(self, chunker):
        page = self._make_page(text="", tables=[])
        chunks = chunker.chunk_page(page)
        assert chunks == []

    def test_year_extracted_into_metadata(self, chunker):
        page = self._make_page(text="Fiscal year 2023 results. " * 5)
        chunks = chunker.chunk_page(page)
        for chunk in chunks:
            assert chunk["metadata"]["year"] == "2023"

    def test_chunk_documents_aggregates_pages(self, chunker):
        pages = [
            self._make_page(text="Page one content. " * 5),
            self._make_page(text="Page two content. " * 5),
        ]
        all_chunks = chunker.chunk_documents(pages)
        assert len(all_chunks) >= 2


# ---------------------------------------------------------------------------
# _make_chunk_id
# ---------------------------------------------------------------------------


class TestMakeChunkId:
    def test_same_inputs_produce_same_id(self):
        id1 = FinancialChunker._make_chunk_id("doc", 1, "prose_0", "some text")
        id2 = FinancialChunker._make_chunk_id("doc", 1, "prose_0", "some text")
        assert id1 == id2

    def test_different_text_produces_different_id(self):
        id1 = FinancialChunker._make_chunk_id("doc", 1, "prose_0", "text A")
        id2 = FinancialChunker._make_chunk_id("doc", 1, "prose_0", "text B")
        assert id1 != id2

    def test_id_is_16_chars(self):
        chunk_id = FinancialChunker._make_chunk_id("doc", 1, "table", "table text")
        assert len(chunk_id) == 16
