import requests


BASE_URL = "http://localhost:7860"


def ask_question(query):

    response = requests.post(
        f"{BASE_URL}/ask",
        json={"query": query}
    )

    print(f"\nQuery: {query}")
    print("Status:", response.status_code)
    print("Response:", response.json())


def test_api():

    response = requests.get(f"{BASE_URL}/")

    print("Root endpoint:", response.status_code)
    print(response.json())

    response = requests.get(f"{BASE_URL}/health")

    print("\nHealth endpoint:", response.status_code)
    print(response.json())

    questions = [
        "What is the delivery fee for small orders?",
        "What is the weather today?",
        "How do I return a damaged item?",
        "Tell me about Zepto Pass membership"
    ]

    for question in questions:
        ask_question(question)


if __name__ == "__main__":
    test_api()