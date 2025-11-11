# =======================================================
#  RBC FAQ SCRAPING AND DATA INSPECTION
# =======================================================
# Use these targets to scrape, inspect, and manage the RBC FAQs dataset.
# - scrape-rbc: Fetches and cleans FAQ pages.
# - inspect-rbc: Generates Markdown summary stats.
# - clean-rbc: Deletes data files (use with caution).
# - logs-rbc: Shows recent scraping logs.
# =======================================================

scrape-rbc:
	@echo "🚀 Starting RBC FAQ scraping..."
	@python src/ingestion/scrape_rbc_faqs.py
	@echo "✅ RBC FAQ scraping complete."

inspect-rbc:
	@echo "🔍 Inspecting cleaned RBC FAQs..."
	@python src/preprocess/inspect_dataset.py data/processed/rbc_faqs.parquet --report
	@echo "📊 Inspection report saved to data/reports/rbc_faqs_report.md"

clean-rbc:
	@echo "🧹 Cleaning up RBC data files..."
	@rm -f data/raw/rbc/*.json data/processed/rbc_faqs.parquet
	@echo "✅ RBC data cleaned."

logs-rbc:
	@echo "📜 Showing latest RBC scraping logs..."
	@tail -n 20 logs/scrape_rbc.log


# =======================================================
#  RAG SYSTEM API AND FRONTEND
# =======================================================
# These targets control your Retrieval-Augmented Generation system.
# - run-api: Starts the backend FastAPI server (with model selection).
# - run-ui:  Launches the web chat UI for user interaction.
# - stop-api: Gracefully terminates any running backend instance.
# - stop-ui:  Gracefully terminates any running frontend instance.
# - auto-clean-memory: Clears OS caches and swap before heavy model loads.
# =======================================================

# Default model (can override at runtime: `make run-api MODEL=llama`)
MODEL ?= phi3

run-api:
	@echo "🚀 Starting RAG API backend using model: $(MODEL)"
	@if [ "$(MODEL)" = "phi3" ]; then \
		export RAG_MODEL="microsoft/Phi-3-mini-4k-instruct"; \
	else \
		export RAG_MODEL="meta-llama/Llama-3.1-8B-Instruct"; \
	fi; \
	PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	@echo "💬 Starting RAG Web Chat UI..."
	PYTHONPATH=src uvicorn frontend.chat_ui:app --host 0.0.0.0 --port 8500 --reload

# Stop backend API process
stop-api:
	@echo "🛑 Stopping RAG API server..."
	@PID=$$(ps aux | grep "uvicorn api.main:app" | grep -v grep | awk '{print $$2}'); \
	if [ -n "$$PID" ]; then \
		echo "🔻 Killing process $$PID"; \
		kill -9 $$PID; \
	else \
		echo "⚠️  No RAG API process found."; \
	fi

# Stop frontend UI process
stop-ui:
	@echo "🛑 Stopping RAG UI server..."
	@PID=$$(ps aux | grep "uvicorn frontend.chat_ui:app" | grep -v grep | awk '{print $$2}'); \
	if [ -n "$$PID" ]; then \
		echo "🔻 Killing process $$PID"; \
		kill -9 $$PID; \
	else \
		echo "⚠️  No RAG UI process found."; \
	fi

# Clear OS caches and swap to reclaim memory before loading large models
auto-clean-memory:
	@echo "🧠 Cleaning system memory and swap to prevent slowdowns..."
	@sudo sync
	@sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
	@sudo swapoff -a && sudo swapon -a
	@echo "✅ Memory and swap cleaned successfully."


# =======================================================
#  DEPENDENCY MANAGEMENT
# =======================================================
# - install: Install all required dependencies.
# - lock: Freeze current environment packages into requirements.lock.
# - sync: Reinstall exact versions from requirements.lock.
# =======================================================

install:
	pip install -r requirements.txt

lock:
	pip freeze > requirements.lock

sync:
	pip install -r requirements.lock
