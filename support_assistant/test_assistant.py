from embedding import load_and_embed_documents
from graph import build_graph


def run_test(graph, query):

    state = {
        "query": query,
        "intent": "",
        "sources": [],
        "answer": "",
        "confidence": 0.0,
        "retrieved_chunks": []
    }

    result = graph.invoke(state)

    print("\nQuery:")
    print(query)

    print("\nIntent:")
    print(result["intent"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(result["sources"])

    print("\nConfidence:")
    print(result["confidence"])

    return result


def main():

    print("Loading support assistant...")

    model, collection = (
        load_and_embed_documents()
    )

    graph = build_graph(
        collection,
        model
    )

    print("\n=== Policy Question ===")

    policy_result = run_test(
        graph,
        "What is the delivery fee for orders below INR 149?"
    )

    assert (
        policy_result["intent"]
        == "policy_question"
    )

    assert policy_result["sources"]

    assert policy_result["answer"].startswith(
        "Based on the retrieved context:"
    )

    print("\n=== General Question ===")

    general_result = run_test(
        graph,
        "What is the weather today?"
    )

    assert (
        general_result["intent"]
        == "general_question"
    )

    assert general_result["sources"] == []

    assert general_result["answer"] == (
        "I can only answer questions about Zepto policies right now."
    )

    print("\nAll support assistant tests passed.")


if __name__ == "__main__":
    main()