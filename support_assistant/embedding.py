import os
import sys
import numpy as np

def load_and_embed_documents():
    """Load documents and create embeddings using simple in-memory storage"""
    
    print("Initializing embedding model...")
    
    # Try to import sentence_transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence_transformers imported successfully")
    except ImportError as e:
        print(f"Error importing sentence_transformers: {e}")
        print("\nPlease run: pip install sentence-transformers")
        sys.exit(1)
    
    # Initialize embedding model
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create a simple in-memory vector store
    print("\nCreating in-memory vector store...")
    
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
            query_emb = np.array(query_embeddings)
            doc_emb = np.array(self.embeddings)
            
            # Compute cosine similarity
            similarities = np.dot(query_emb, doc_emb.T) / (
                np.linalg.norm(query_emb, axis=1)[:, np.newaxis] * 
                np.linalg.norm(doc_emb, axis=1)[np.newaxis, :]
            )
            
            results = []
            for sim_row in similarities:
                top_indices = np.argsort(sim_row)[-n_results:][::-1]
                results.append({
                    'documents': [[self.documents[i] for i in top_indices]],
                    'ids': [[self.ids[i] for i in top_indices]],
                    'metadatas': [[self.metadatas[i] for i in top_indices]],
                    'distances': [[1 - sim_row[i] for i in top_indices]]
                })
            
            return results[0] if results else {
                'documents': [[]],
                'ids': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
        
        def count(self):
            return len(self.documents)
    
    collection = SimpleVectorStore()
    print("✓ In-memory vector store created")
    
    # Load documents
    print("\nLoading documents...")
    docs_dir = "docs"
    
    if not os.path.exists(docs_dir):
        print(f"ERROR: Documents directory '{docs_dir}' not found!")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    documents = []
    metadatas = []
    ids = []
    
    for i in range(1, 9):
        doc_id = f"doc_{i:02d}"
        filepath = os.path.join(docs_dir, f"{doc_id}.txt")
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            documents.append(content)
            metadatas.append({"doc_id": doc_id, "source": filepath})
            ids.append(doc_id)
            print(f"✓ Loaded {doc_id}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    if not documents:
        print("ERROR: No documents loaded!")
        sys.exit(1)
    
    print(f"\nLoaded {len(documents)} documents")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    try:
        embeddings = model.encode(documents, show_progress_bar=True)
        print(f"✓ Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        sys.exit(1)
    
    # Add to collection
    print("\nAdding to vector store...")
    try:
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✓ Added {collection.count()} documents to vector store")
    except Exception as e:
        print(f"Error adding to vector store: {e}")
        sys.exit(1)
    
    print("\n=== EMBEDDING COMPLETE ===")
    print(f"Vector store contains {collection.count()} documents")
    
    return model, collection