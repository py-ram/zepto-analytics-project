FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install with specific version pins to avoid conflicts
RUN pip install --no-cache-dir \
    huggingface-hub==0.16.4 \
    sentence-transformers==2.2.2 \
    transformers==4.34.0 \
    tokenizers==0.14.1 \
    chromadb==0.4.15 \
    langgraph==0.0.20 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.4.2 \
    numpy==1.24.3 \
    pandas==2.0.3

# Copy the rest of the application
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]