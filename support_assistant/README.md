# Support Assistant

## Overview

This module is a small support assistant that answers questions related to Zepto policies.

It uses sentence-transformers to convert the policy documents and user question into embeddings. The application then compares the embeddings using cosine similarity and retrieves the most relevant documents.

## How it works

1. Policy documents are loaded from the `docs` folder.
2. Each document is converted into an embedding.
3. A user question is converted into an embedding.
4. Cosine similarity is used to find relevant documents.
5. The top matching documents are returned as sources.
6. Policy questions are answered using the retrieved text.

## Main Files

- `embedding.py` - Loads documents and creates embeddings.
- `graph.py` - Handles intent classification and retrieval.
- `main.py` - FastAPI application.
- `test_assistant.py` - Tests the assistant without the API.
- `test_api.py` - Tests the FastAPI endpoints.

## Run the Assistant

Install dependencies:

```bash
pip install -r requirements.txt