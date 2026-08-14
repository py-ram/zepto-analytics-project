import os
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"
DB_PATH = "db"


def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """
    Split a document into small chunks.

    The policy documents are short, so a simple fixed-size
    chunking approach is enough for this assignment.
    """

    text = text.strip()

    if not text:
        return []

    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def load_documents(docs_dir: str = "docs") -> Tuple[List[str], List[str], List[dict]]:
    """Load the eight policy documents from the docs folder."""

    documents = []
    ids = []
    metadatas = []

    for i in range(1, 9):

        doc_id = f"doc_{i:02d}"
        file_path = os.path.join(
            docs_dir,
            f"{doc_id}.txt"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required document not found: {file_path}"
            )

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):

            chunk_id = f"{doc_id}_chunk_{chunk_number + 1}"

            documents.append(chunk)
            ids.append(chunk_id)

            metadatas.append({
                "doc_id": doc_id,
                "source": file_path,
                "chunk": chunk_number + 1
            })

    return documents, ids, metadatas


def load_and_embed_documents():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading policy documents...")

    documents, ids, metadatas = load_documents()

    print(f"Loaded {len(documents)} document chunks.")

    print("Creating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=False
    ).tolist()

    # ChromaDB stores the vectors locally.
    os.makedirs(DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    # Recreate the collection so the local database
    # always matches the current policy documents.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"ChromaDB collection '{COLLECTION_NAME}' "
        f"contains {collection.count()} chunks."
    )

    return model, collection