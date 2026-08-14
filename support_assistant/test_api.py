import requests


BASE_URL = "http://127.0.0.1:7860"


def test_api():

    response = requests.get(
        f"{BASE_URL}/health"
    )

    print("Health:")
    print(response.status_code)
    print(response.json())

    questions = [
        "What is the delivery fee for orders below INR 149?",
        "What is the weather today?"
    ]

    for question in questions:

        response = requests.post(
            f"{BASE_URL}/ask",
            json={
                "query": question
            }
        )

        print("\nQuestion:")
        print(question)

        print("\nStatus:")
        print(response.status_code)

        print("\nJSON response:")
        print(response.json())


if __name__ == "__main__":
    test_api()