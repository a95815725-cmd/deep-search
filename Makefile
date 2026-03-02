.PHONY: install dev run mcp format lint check test help

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  install     Install production dependencies (uv sync)"
	@echo "  dev         Install all dependencies including dev tools"
	@echo "  run         Launch the Streamlit UI"
	@echo "  mcp         Start the MCP server (stdio)"
	@echo "  format      Format code with ruff"
	@echo "  lint        Lint code with ruff"
	@echo "  test        Run the unit test suite"
	@echo "  check       Run all pre-commit hooks on every file"
	@echo ""

# ── Dependencies ──────────────────────────────────────────────────────────────
install:
	uv sync

dev:
	uv sync --extra dev
	uv run pre-commit install

# ── Run ───────────────────────────────────────────────────────────────────────
run:
	uv run streamlit run app.py

mcp:
	uv run python mcp_server.py

# ── Code quality ──────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

format:
	uv run ruff format .

lint:
	uv run ruff check .

check:
	uv run pre-commit run --all-files
