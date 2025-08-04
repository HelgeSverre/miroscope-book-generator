# Makefile for Parallel Book Generator

# Default target - runs everything
.PHONY: all
all: install format run

# Install dependencies from requirements.txt and pyproject.toml
.PHONY: install
install:
	@echo "📥 Installing dependencies..."
	@if [ -f "pyproject.toml" ]; then \
		echo "   Found pyproject.toml - using uv sync..."; \
		uv sync; \
	elif [ -f "requirements.txt" ]; then \
		echo "   Found requirements.txt - using uv pip install..."; \
		uv pip install -r requirements.txt; \
	else \
		echo "   ❌ No requirements.txt or pyproject.toml found!"; \
		exit 1; \
	fi

# Install/sync dependencies with uv (alias for install)
.PHONY: sync
sync: install

# Install dependencies using pip (traditional method)
.PHONY: pip-install
pip-install:
	@echo "📦 Installing dependencies with pip..."
	@if [ -f "requirements.txt" ]; then \
		pip install -r requirements.txt; \
	else \
		echo "   ❌ No requirements.txt found!"; \
		exit 1; \
	fi

# Update dependencies
.PHONY: update
update:
	@echo "🔄 Updating dependencies..."
	@if [ -f "pyproject.toml" ]; then \
		uv sync --upgrade; \
	elif [ -f "requirements.txt" ]; then \
		uv pip install --upgrade -r requirements.txt; \
	fi

# Format code with ruff
.PHONY: format
format:
	@echo "🎨 Formatting code with ruff..."
	uv run ruff format .

# Run the parallel book generator
.PHONY: run
run:
	@echo "🚀 Running parallel book generator..."
	@echo "📚 Reading books from config.yml..."
	uv run python run_parallel.py


# Check code with ruff (without fixing)
.PHONY: check
check:
	@echo "🔍 Checking code with ruff..."
	uv run ruff check .

# Fix code issues with ruff
.PHONY: fix
fix:
	@echo "🔧 Fixing code issues with ruff..."
	uv run ruff check --fix .

# Clean generated books
.PHONY: clean
clean:
	@echo "🧹 Cleaning generated books..."
	rm -rf generated_books/

# Clean logs
.PHONY: clean-logs
clean-logs:
	@echo "📜 Cleaning logs..."
	rm -rf logs/

# Clean all generated files and logs
.PHONY: clean-all
clean-all: clean clean-logs
	@echo "✨ All generated files and logs cleaned!"

# Development workflow - format and run
.PHONY: dev
dev: format run

# Full workflow - install, format, and run
.PHONY: full
full: install format run

# Show available targets
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make all     - Run install, format, and run (default)"
	@echo "  make install - Install dependencies (auto-detects pyproject.toml or requirements.txt)"
	@echo "  make sync    - Alias for install"
	@echo "  make update  - Update dependencies to latest versions"
	@echo "  make pip-install - Install with traditional pip (requirements.txt only)"
	@echo "  make format  - Format code with ruff"
	@echo "  make run     - Run the parallel book generator"
	@echo "  make check   - Check code with ruff (no fixes)"
	@echo "  make fix     - Fix code issues with ruff"
	@echo "  make clean   - Remove generated books"
	@echo "  make clean-logs - Remove log files"
	@echo "  make clean-all - Remove all generated files and logs"
	@echo "  make dev     - Format and run (for development)"
	@echo "  make full    - Full workflow: install, format, run"
	@echo "  make help    - Show this help message"