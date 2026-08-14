from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


app = FastAPI(
    title="Zepto Support Assistant",
    description="Support assistant for Zepto policy questions",
    version="1.0.0"
)


model = None
collection = None
graph = None


class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="Customer question"
    )


class AnswerSchema(BaseModel):

    answer: str

    sources: list[str] = Field(
        default_factory=list
    )

    confidence: float = Field(
        ge=0,
        le=1
    )


@app.on_event("startup")
async def startup_event():

    global model
    global collection
    global graph

    print("\nLoading support assistant...")

    try:

        from embedding import load_and_embed_documents
        from graph import build_graph

        model, collection = (
            load_and_embed_documents()
        )

        graph = build_graph(
            collection,
            model
        )

        print("Support assistant is ready.")

    except Exception as error:

        print(
            f"Could not load support assistant: {error}"
        )


@app.get("/")
async def root():

    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():

    if graph is None:

        return {
            "status": "degraded",
            "message": "Support assistant is not loaded"
        }

    return {
        "status": "ok",
        "message": "Support assistant is ready"
    }


@app.post(
    "/ask",
    response_model=AnswerSchema
)
async def ask(request: QueryRequest):

    if graph is None:

        raise HTTPException(
            status_code=503,
            detail="Support assistant is not loaded"
        )

    state = {
        "query": request.query,
        "intent": "",
        "sources": [],
        "answer": "",
        "confidence": 0.0,
        "retrieved_chunks": []
    }

    try:

        result = graph.invoke(state)

        return AnswerSchema(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


if __name__ == "__main__":

    print(
        "\nStarting server at "
        "http://127.0.0.1:7860"
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=7860
    )