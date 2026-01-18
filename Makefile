.PHONY: install train run dev clean help

# Default target
help:
	@echo "🏠 House Price Prediction - Available Commands"
	@echo "================================================"
	@echo "make install    - Install dependencies with uv"
	@echo "make train      - Train the ML model"
	@echo "make run        - Run Streamlit app"
	@echo "make dev        - Run app with auto-reload"
	@echo "make clean      - Clean cache and temp files"
	@echo "make all        - Install, train, and run"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	uv sync

# Train model
train:
	@echo "🧠 Training model..."
	uv run python src/train_model.py

# Run Streamlit app
run:
	@echo "🚀 Starting Streamlit app..."
	uv run streamlit run app.py

# Run with auto-reload (development mode)
dev:
	@echo "🔄 Starting Streamlit in dev mode..."
	uv run streamlit run app.py --server.runOnSave true

# Clean cache files
clean:
	@echo "🧹 Cleaning cache..."
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf src/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "✅ Clean complete!"

# Full setup: install, train, and run
all: install train run
