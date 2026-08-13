from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import sys
import socket

# Create FastAPI app
app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based support assistant for Zepto policies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
model = None
collection = None
graph = None

@app.on_event("startup")
async def startup_event():
    """Load models at startup"""
    global model, collection, graph
    
    print("\nLoading models...")
    try:
        from embedding import load_and_embed_documents
        from graph import build_graph
        
        model, collection = load_and_embed_documents()
        graph = build_graph(collection, model)
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load models: {e}")
        print("API will start but /ask endpoint may not work")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Your question")
    
    class Config:
        schema_extra = {
            "example": {"query": "What is the delivery fee?"}
        }

class AnswerSchema(BaseModel):
    answer: str
    sources: list = []
    confidence: float = 0.0

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check"""
    return HealthResponse(
        status="ok" if graph else "degraded",
        message="Service is healthy" if graph else "Model not loaded"
    )

@app.get("/test", tags=["General"])
async def test():
    """Test endpoint"""
    return {"message": "API is working!"}

@app.post("/ask", response_model=AnswerSchema, tags=["Q&A"])
async def ask(request: QueryRequest):
    """Ask a question"""
    
    if graph is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        initial_state = {
            "query": request.query,
            "intent": "",
            "sources": [],
            "answer": "",
            "confidence": 0.0,
            "retrieved_chunks": []
        }
        
        result = graph.invoke(initial_state)
        
        return AnswerSchema(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_available_port():
    """Find an available port"""
    for port in [7860, 7861, 7862, 8000, 8080, 5000]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except OSError:
            continue
    return 7860

if __name__ == "__main__":
    port = find_available_port()
    
    print("\n" + "="*50)
    print("Zepto Support Assistant")
    print("="*50)
    print(f"\nServer starting on: http://127.0.0.1:{port}")
    print(f"Swagger UI: http://127.0.0.1:{port}/docs")
    print(f"ReDoc: http://127.0.0.1:{port}/redoc")
    print("\nPress Ctrl+C to stop")
    print("="*50 + "\n")
    
    # Use 127.0.0.1 instead of 0.0.0.0 to avoid getaddrinfo issues
    uvicorn.run(app, host="127.0.0.1", port=port)