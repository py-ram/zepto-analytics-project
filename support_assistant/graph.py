from typing import TypedDict, List
from pydantic import BaseModel, Field
import os

class GraphState(TypedDict):
    query: str
    intent: str
    sources: List[str]
    answer: str
    confidence: float
    retrieved_chunks: List[str]

class AnswerSchema(BaseModel):
    answer: str = Field(description="The answer to the query")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs")
    confidence: float = Field(description="Confidence score", ge=0, le=1)

# Mock LLM toggle
MOCK_LLM = os.getenv('MOCK_LLM', '1') != '0'
print(f"MOCK_LLM mode: {'enabled (mock)' if MOCK_LLM else 'disabled (real LLM)'}")

# Keyword heuristic
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", 
    "tracking", "cancel", "gift card", "support hours",
    "damaged", "missing", "order", "replacement"
]

def classify_intent(state: GraphState) -> GraphState:
    """Classify query intent"""
    query_lower = state['query'].lower()
    
    matched_keywords = [kw for kw in POLICY_KEYWORDS if kw in query_lower]
    
    if matched_keywords:
        state['intent'] = 'policy_question'
    else:
        state['intent'] = 'general_question'
    
    return state

def retrieve_and_answer(state: GraphState, collection, model) -> GraphState:
    """Retrieve and answer policy questions"""
    
    try:
        query_embedding = model.encode([state['query']])
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        
        retrieved_chunks = results['documents'][0]
        chunk_ids = results['ids'][0]
        
        state['retrieved_chunks'] = retrieved_chunks
        state['sources'] = chunk_ids
        
        if retrieved_chunks:
            top_chunk_snippet = retrieved_chunks[0][:200]
            state['answer'] = f"Based on the retrieved context: {top_chunk_snippet}"
        else:
            state['answer'] = "No relevant policy information found."
    except Exception as e:
        state['retrieved_chunks'] = []
        state['sources'] = []
        state['answer'] = f"Error during retrieval: {str(e)}"
    
    state['confidence'] = 1.0 if state['retrieved_chunks'] else 0.0
    return state

def direct_answer(state: GraphState) -> GraphState:
    """Answer general questions"""
    state['answer'] = "I can only answer questions about Zepto policies right now."
    state['sources'] = []
    state['confidence'] = 1.0
    return state

class SimpleGraph:
    """Simple graph implementation without langgraph dependency"""
    
    def __init__(self, collection, model):
        self.collection = collection
        self.model = model
    
    def invoke(self, state):
        """Process state through the graph"""
        # Classify intent
        state = classify_intent(state)
        
        # Route based on intent
        if state['intent'] == 'policy_question':
            state = retrieve_and_answer(state, self.collection, self.model)
        else:
            state = direct_answer(state)
        
        return state

def build_graph(collection, model):
    """Build and return graph"""
    print("Building graph...")
    return SimpleGraph(collection, model)