# Support Assistant

## Overview

This module provides a small support assistant for answering questions related to Zepto policies.

The application uses sentence-transformers to create embeddings for the policy documents. When a user asks a question, the question is also converted into an embedding and compared with the stored document embeddings.

The most relevant documents are then returned as the source for the answer.

## Project Structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── embedding.py
├── graph.py
├── main.py
├── test_assistant.py
├── test_api.py
├── requirements.txt
└── Dockerfile



How It Works
Policy documents are loaded from the docs folder.
Sentence-transformers creates embeddings for the documents.
The user's question is converted into an embedding.
Cosine similarity is used to compare the question with the documents.
The most relevant documents are retrieved.
The retrieved document information is returned as the answer source.
Main Files
embedding.py

Loads the policy documents and creates their embeddings.

It also contains the simple in-memory vector store used for similarity search.

graph.py

Handles basic intent classification and routes policy questions to the retrieval step.

main.py

Runs the FastAPI application and provides the API endpoints.

test_assistant.py

Runs a few local tests against the assistant without starting the API.

test_api.py

Tests the FastAPI endpoints.

Running Locally

Install the required packages:

pip install -r requirements.txt

Start the application:

python main.py

The API documentation can then be opened from the URL printed by the application.

For example:

http://127.0.0.1:7860/docs
Testing

Run the local assistant test:

python test_assistant.py

Run the API test while the server is running:

python test_api.py
Limitations

This is a basic prototype.

The current version uses keyword-based intent classification and an in-memory vector store. It does not use a separate production vector database.

The retrieved document text is used directly when preparing the response, so the assistant is intended for demonstrating the retrieval workflow rather than serving as a production support system.
