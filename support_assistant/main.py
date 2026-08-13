from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import socket


app = FastAPI(
    title="Zepto Support Assistant",
    description="Support assistant for Zepto policy questions",
    version="1.0.0"
)

model = None
collection = None
graph = None


@app.on_event("startup")
async def startup_event():

    global model, collection, graph

    print("\nLoading support documents...")

    try:
        from embedding import load_and_embed_documents
        from graph import build_graph

        model, collection = load_and_embed_documents()
        graph = build_graph(collection, model)

        print("Support assistant loaded.")

    except Exception as e:
        print("Could not load support assistant:", e)


class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="Your question"
    )


class AnswerSchema(BaseModel):

    answer: str
    sources: list[str] = []
    confidence: float = 0.0


class HealthResponse(BaseModel):

    status: str
    message: str


@app.get("/")
async def root():

    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():

    if graph is not None:
        return HealthResponse(
            status="ok",
            message="Support assistant is ready"
        )

    return HealthResponse(
        status="degraded",
        message="Support assistant is not loaded"
    )


@app.get("/test")
async def test():

    return {"message": "API is working"}


@app.post("/ask", response_model=AnswerSchema)
async def ask(request: QueryRequest):

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Support assistant is not loaded"
        )

    try:

        state = {
            "query": request.query,
            "intent": "",
            "sources": [],
            "answer": "",
            "confidence": 0.0,
            "retrieved_chunks": []
        }

        result = graph.invoke(state)

        return AnswerSchema(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


def find_available_port():

    for port in [7860, 7861, 7862, 8000, 8080]:

        sock = socket.socket()

        try:
            sock.bind(("127.0.0.1", port))
            sock.close()
            return port

        except OSError:
            sock.close()

    return 7860


if __name__ == "__main__":

    port = find_available_port()

    print(f"\nStarting server on http://127.0.0.1:{port}")
    print(f"API docs: http://127.0.0.1:{port}/docs")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port
    )