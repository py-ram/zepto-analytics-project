import os
from typing import TypedDict, List

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the policy information provided below.

TASK:
Answer the customer's question using the retrieved Zepto policy context.

FORMAT:
Return a JSON object with:
- answer: string
- sources: list of document or chunk IDs
- confidence: number between 0 and 1

LENGTH:
Keep the answer short and directly relevant to the customer's question.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, timings, refunds or other details.

FEW-SHOT EXAMPLE:

Question:
"What is the delivery fee for an order below INR 149?"

Context:
"Standard delivery is free on orders over INR 149; orders below this
threshold incur a flat INR 25 delivery fee."

Expected answer:
{
  "answer": "Orders below INR 149 have a flat INR 25 delivery fee.",
  "sources": ["doc_01_chunk_1"],
  "confidence": 1.0
}

Now answer the following question using only the supplied context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""


class GraphState(TypedDict):

    query: str
    intent: str
    sources: List[str]
    answer: str
    confidence: float
    retrieved_chunks: List[str]


class AnswerSchema(BaseModel):

    answer: str

    sources: List[str] = Field(
        default_factory=list
    )

    confidence: float = Field(
        ge=0,
        le=1
    )


def classify_intent(state: GraphState) -> GraphState:

    query = state["query"].lower()

    if any(
        keyword in query
        for keyword in POLICY_KEYWORDS
    ):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state


def retrieve_and_answer(
    state: GraphState,
    collection,
    model
) -> GraphState:

    query_embedding = model.encode(
        [state["query"]]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    chunks = results.get(
        "documents",
        [[]]
    )[0]

    source_ids = results.get(
        "ids",
        [[]]
    )[0]

    state["retrieved_chunks"] = chunks
    state["sources"] = source_ids

    if not chunks:

        state["answer"] = (
            "No relevant policy information was found."
        )

        state["confidence"] = 0.0

        return state

    top_chunk = chunks[0]

    if MOCK_LLM:

        state["answer"] = (
            "Based on the retrieved context: "
            + top_chunk[:200]
        )

        state["confidence"] = 1.0

        return state

    # Optional real-LLM path
    answer = generate_real_llm_answer(
        question=state["query"],
        context=chunks,
        source_ids=source_ids
    )

    state["answer"] = answer.answer
    state["sources"] = answer.sources
    state["confidence"] = answer.confidence

    return state


def direct_answer(state: GraphState) -> GraphState:

    if MOCK_LLM:

        state["answer"] = (
            "I can only answer questions about Zepto policies right now."
        )

        state["sources"] = []
        state["confidence"] = 1.0

        return state

    # Optional real-LLM path
    answer = generate_real_llm_answer(
        question=state["query"],
        context=[],
        source_ids=[]
    )

    state["answer"] = answer.answer
    state["sources"] = []
    state["confidence"] = answer.confidence

    return state


def route_query(state: GraphState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def generate_real_llm_answer(
    question: str,
    context: List[str],
    source_ids: List[str]
) -> AnswerSchema:
    """
    Optional real-LLM extension.

    This function is never called in the default MOCK_LLM=1 mode.
    """

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when MOCK_LLM=0."
        )

    client = OpenAI(
        api_key=api_key
    )

    context_text = "\n\n".join(context)

    prompt = PROMPT_TEMPLATE.format(
        question=question,
        context=context_text
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    last_error = None

    for attempt in range(3):

        try:

            response = client.chat.completions.create(
                model=os.getenv(
                    "OPENAI_MODEL",
                    "gpt-4o-mini"
                ),
                messages=messages,
                temperature=0
            )

            raw_answer = response.choices[0].message.content

            parsed = AnswerSchema.model_validate_json(
                raw_answer
            )

            return parsed

        except Exception as error:

            last_error = error

            messages.append({
                "role": "user",
                "content": (
                    "Your previous response did not match the required "
                    "JSON schema. Return only valid JSON containing "
                    "answer, sources and confidence."
                )
            })

    return AnswerSchema(
        answer=(
            "ERROR: The optional real-LLM response "
            "could not be validated."
        ),
        sources=source_ids,
        confidence=0.0
    )


def build_graph(
    collection,
    model
):

    workflow = StateGraph(GraphState)

    workflow.add_node(
        "classify_intent",
        classify_intent
    )

    workflow.add_node(
        "retrieve_and_answer",
        lambda state: retrieve_and_answer(
            state,
            collection,
            model
        )
    )

    workflow.add_node(
        "direct_answer",
        direct_answer
    )

    workflow.set_entry_point(
        "classify_intent"
    )

    workflow.add_conditional_edges(
        "classify_intent",
        route_query,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    workflow.add_edge(
        "retrieve_and_answer",
        END
    )

    workflow.add_edge(
        "direct_answer",
        END
    )

    return workflow.compile()