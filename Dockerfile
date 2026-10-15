FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (libgl1 and libglib2.0-0 are often needed for OpenCV/FAISS/Torch vision)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY continual_rag/ continual_rag/
COPY scripts/ scripts/
COPY tests/ tests/

# Install the package with embedding dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[embed]"

# Default command
CMD ["python", "scripts/run_benchmark.py", "--regime", "base5-same"]
