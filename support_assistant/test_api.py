import requests
import json

def test_api():
    """Test the FastAPI endpoints"""
    
    base_url = "http://localhost:7860"
    
    # Test root endpoint
    print("Testing root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test policy question
    print("\n" + "="*50)
    print("Testing policy question...")
    query = "What is the delivery fee for small orders?"
    response = requests.post(
        f"{base_url}/ask",
        json={"query": query}
    )
    print(f"Query: {query}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test general question
    print("\n" + "="*50)
    print("Testing general question...")
    query = "What is the weather today?"
    response = requests.post(
        f"{base_url}/ask",
        json={"query": query}
    )
    print(f"Query: {query}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test another policy question
    print("\n" + "="*50)
    print("Testing another policy question...")
    query = "How do I return a damaged item?"
    response = requests.post(
        f"{base_url}/ask",
        json={"query": query}
    )
    print(f"Query: {query}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test membership question
    print("\n" + "="*50)
    print("Testing membership question...")
    query = "Tell me about Zepto Pass membership"
    response = requests.post(
        f"{base_url}/ask",
        json={"query": query}
    )
    print(f"Query: {query}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    test_api()
