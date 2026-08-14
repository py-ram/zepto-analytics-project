# Support Assistant

## Overview

This module implements a small retrieval-based support assistant for Zepto policy questions.

The application uses local sentence-transformer embeddings, ChromaDB for vector storage, LangGraph for query routing and FastAPI for the API layer.

The default execution mode uses deterministic mock responses, so no LLM API key or network access is required.

## Architecture

```text
Policy Documents
      |
      v
Document Loading + Chunking
      |
      v
all-MiniLM-L6-v2
      |
      v
ChromaDB
      |
      v
LangGraph StateGraph
      |
      v
classify_intent
      |
      +----------------------+
      |                      |
      v                      v
policy_question       general_question
      |                      |
      v                      v
retrieve_and_answer    direct_answer
      |
      v
MOCK_LLM
      |
      v
Pydantic Response
      |
      v
FastAPI /ask