from typing import TypedDict, List
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    query: str
    intent: str
    sources: List[str]
    answer: str
    confidence: float
    retrieved_chunks: List[str]


class AnswerSchema(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
    "damaged",
    "missing",
    "order",
    "replacement"
]


def classify_intent(state: GraphState) -> GraphState:

    query = state["query"].lower()

    if any(word in query for word in POLICY_KEYWORDS):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state


def retrieve_and_answer(
    state: GraphState,
    collection,
    model
) -> GraphState:

    try:
        query_embedding = model.encode([state["query"]])

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )

        chunks = results["documents"][0]
        sources = results["ids"][0]

        state["retrieved_chunks"] = chunks
        state["sources"] = sources

        if chunks:
            state["answer"] = (
                "Based on the available policy information: "
                + chunks[0][:200]
            )
            state["confidence"] = 1.0
        else:
            state["answer"] = "No relevant policy information found."
            state["confidence"] = 0.0

    except Exception as e:
        state["retrieved_chunks"] = []
        state["sources"] = []
        state["answer"] = f"Retrieval error: {e}"
        state["confidence"] = 0.0

    return state


def direct_answer(state: GraphState) -> GraphState:

    state["answer"] = (
        "I can only answer questions about Zepto policies right now."
    )

    state["sources"] = []
    state["confidence"] = 1.0

    return state


class SimpleGraph:

    def __init__(self, collection, model):
        self.collection = collection
        self.model = model

    def invoke(self, state):

        state = classify_intent(state)

        if state["intent"] == "policy_question":
            return retrieve_and_answer(
                state,
                self.collection,
                self.model
            )

        return direct_answer(state)


def build_graph(collection, model):

    return SimpleGraph(collection, model)