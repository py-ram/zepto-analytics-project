# support_assistant/test_assistant.py
import os
import json
from embedding import load_and_embed_documents
from graph import build_graph, AnswerSchema

def test_assistant():
    """Test the support assistant without FastAPI"""
    
    print("="*50)
    print("TESTING SUPPORT ASSISTANT")
    print("="*50)
    
    # Load models
    print("\n[1] Loading models and embeddings...")
    model, collection = load_and_embed_documents()
    
    # Build graph
    print("\n[2] Building LangGraph...")
    graph = build_graph(collection, model)
    
    # Test queries
    test_queries = [
        {
            "query": "What is the delivery fee for small orders?",
            "expected_intent": "policy_question"
        },
        {
            "query": "What's the weather like today?",
            "expected_intent": "general_question"
        },
        {
            "query": "How do I return a damaged item?",
            "expected_intent": "policy_question"
        },
        {
            "query": "Tell me about Zepto Pass membership",
            "expected_intent": "policy_question"
        }
    ]
    
    print("\n[3] Testing queries...")
    results = []
    
    for test in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{test['query']}'")
        print(f"Expected intent: {test['expected_intent']}")
        
        initial_state = {
            "query": test["query"],
            "intent": "",
            "sources": [],
            "answer": "",
            "confidence": 0.0,
            "retrieved_chunks": []
        }
        
        result = graph.invoke(initial_state)
        
        # Validate response
        response = AnswerSchema(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence']
        )
        
        print(f"\nActual intent: {result['intent']}")
        print(f"Response: {response.model_dump_json(indent=2)}")
        
        # Check if intent matches expectation
        intent_match = result['intent'] == test['expected_intent']
        print(f"Intent match: {'✓' if intent_match else '✗'}")
        
        results.append({
            'query': test['query'],
            'expected_intent': test['expected_intent'],
            'actual_intent': result['intent'],
            'intent_match': intent_match,
            'response': response
        })
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    all_passed = all(r['intent_match'] for r in results)
    print(f"All intents matched: {'✓' if all_passed else '✗'}")
    print(f"Tests passed: {sum(r['intent_match'] for r in results)}/{len(results)}")
    
    # Print detailed results
    print("\nDetailed Results:")
    for i, r in enumerate(results, 1):
        print(f"\nTest {i}:")
        print(f"  Query: {r['query']}")
        print(f"  Intent: {r['actual_intent']} (expected: {r['expected_intent']})")
        print(f"  Answer: {r['response'].answer[:100]}...")
        print(f"  Sources: {r['response'].sources}")
        print(f"  Confidence: {r['response'].confidence}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_assistant()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()