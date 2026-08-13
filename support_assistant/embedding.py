import os
import sys
import numpy as np


def load_and_embed_documents():

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed.")
        sys.exit(1)

    print("Loading embedding model...")

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print("Could not load embedding model:", e)
        sys.exit(1)

    class SimpleVectorStore:

        def __init__(self):
            self.documents = []
            self.embeddings = []
            self.ids = []
            self.metadatas = []

        def add(self, embeddings, documents, metadatas, ids):
            self.embeddings.extend(embeddings)
            self.documents.extend(documents)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)

        def query(self, query_embeddings, n_results=3):

            query = np.array(query_embeddings)
            docs = np.array(self.embeddings)

            # Calculate cosine similarity
            similarity = np.dot(query, docs.T) / (
                np.linalg.norm(query, axis=1)[:, None]
                * np.linalg.norm(docs, axis=1)[None, :]
            )

            results = []

            for row in similarity:
                top_indices = np.argsort(row)[-n_results:][::-1]

                results.append({
                    "documents": [
                        [self.documents[i] for i in top_indices]
                    ],
                    "ids": [
                        [self.ids[i] for i in top_indices]
                    ],
                    "metadatas": [
                        [self.metadatas[i] for i in top_indices]
                    ],
                    "distances": [
                        [1 - row[i] for i in top_indices]
                    ]
                })

            return results[0]

        def count(self):
            return len(self.documents)

    collection = SimpleVectorStore()

    docs_dir = "docs"

    if not os.path.exists(docs_dir):
        print(f"Documents folder not found: {docs_dir}")
        sys.exit(1)

    documents = []
    metadatas = []
    ids = []

    # Load the eight policy documents
    for i in range(1, 9):

        doc_id = f"doc_{i:02d}"
        file_path = os.path.join(
            docs_dir,
            f"{doc_id}.txt"
        )

        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()

            documents.append(content)
            ids.append(doc_id)
            metadatas.append({
                "doc_id": doc_id,
                "source": file_path
            })

        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    if not documents:
        print("No documents found.")
        sys.exit(1)

    print(f"Loaded {len(documents)} documents.")

    # Create embeddings
    try:
        embeddings = model.encode(documents)
    except Exception as e:
        print("Could not create embeddings:", e)
        sys.exit(1)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Vector store contains {collection.count()} documents.")

    return model, collection