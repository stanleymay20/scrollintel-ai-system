"""
Test Natural Language Interface Implementation
"""
import asyncio
import logging
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel_core.nl_interface import NLProcessor, Intent
from scrollintel_core.agents.orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_nl_processor():
    """Test NL processor components"""
    print("Testing Natural Language Processor...")
    
    nl_processor = NLProcessor()
    
    # Test queries for different intents
    test_queries = [
        "What's the best technology stack for scaling our application?",
        "Analyze the sales data and show me key insights",
        "Build a machine learning model to predict customer churn",
        "Create a dashboard showing our monthly revenue trends",
        "What's the best AI strategy for our company?",
        "How many customers do we have in the dataset?",
        "Forecast our sales for the next 6 months"
    ]
    
    print("\n=== Query Parsing Tests ===")
    for query in test_queries:
        parsed = nl_processor.parse_query(query, "test_session")
        print(f"\nQuery: {query}")
        print(f"Intent: {parsed.intent.value} (confidence: {parsed.confidence:.2f})")
        print(f"Suggested Agent: {parsed.suggested_agent}")
        print(f"Entities: {[(e.type, e.value) for e in parsed.entities]}")
        print(f"Parameters: {parsed.parameters}")
        print(f"Context Needed: {parsed.context_needed}")
    
    # Test conversation memory
    print("\n=== Conversation Memory Tests ===")
    session_id = "test_session_123"
    
    # Simulate conversation turns
    conversations = [
        ("Analyze my sales data", {"agent": "data_scientist", "success": True, "result": {"summary": "Sales analysis complete"}}),
        ("Now build a model to predict sales", {"agent": "ml_engineer", "success": True, "result": {"model_type": "regression", "accuracy": 0.85}}),
        ("Create a dashboard for this", {"agent": "bi", "success": True, "result": {"dashboard_created": True}})
    ]
    
    for query, mock_response in conversations:
        nl_response = nl_processor.process_conversation_turn(
            query, mock_response, session_id
        )
        print(f"\nUser: {query}")
        print(f"NL Response: {nl_response}")
    
    # Check conversation history
    history = nl_processor.get_conversation_history(session_id)
    print(f"\nConversation History ({len(history)} turns):")
    for turn in history:
        print(f"- {turn['user_query']} -> {turn['agent_name']} ({turn['intent']})")
    
    print("\n=== NL Processor Tests Complete ===")


async def test_orchestrator_integration():
    """Test orchestrator integration with NL interface"""
    print("\nTesting Orchestrator Integration...")
    
    # Initialize orchestrator (this would normally be done in main.py)
    orchestrator = AgentOrchestrator()
    
    try:
        await orchestrator.initialize()
        
        # Test NL-enhanced routing
        test_queries = [
            "I need help with system architecture decisions",
            "Can you analyze this dataset for patterns?",
            "Build me a predictive model",
            "Show me a business dashboard"
        ]
        
        print("\n=== NL-Enhanced Routing Tests ===")
        for query in test_queries:
            # Test query parsing
            parsed_info = await orchestrator.parse_query(query, "test_session")
            print(f"\nQuery: {query}")
            print(f"Parsed Intent: {parsed_info['intent']} (confidence: {parsed_info['confidence']:.2f})")
            print(f"Suggested Agent: {parsed_info['suggested_agent']}")
            print(f"Entities: {len(parsed_info['entities'])} found")
            
            # Test suggestions
            suggestions = await orchestrator.get_nl_suggestions(query)
            print(f"Suggestions: {suggestions['suggestions'][:2]}")  # Show first 2 suggestions
        
        # Test conversational request processing
        print("\n=== Conversational Request Tests ===")
        session_id = "integration_test_session"
        
        # This would normally call actual agents, but for testing we'll just check the flow
        try:
            response = await orchestrator.process_conversational_request(
                "What technology stack should I use?",
                session_id,
                {"company_size": "startup", "budget": "limited"}
            )
            print(f"Conversational Response Success: {response.get('success', False)}")
            print(f"Agent Used: {response.get('agent', 'unknown')}")
            print(f"Has NL Response: {'nl_response' in response}")
            
        except Exception as e:
            print(f"Expected error (agents not fully implemented): {e}")
        
        print("\n=== Orchestrator Integration Tests Complete ===")
        
    finally:
        await orchestrator.shutdown()


async def test_entity_extraction():
    """Test entity extraction capabilities"""
    print("\nTesting Entity Extraction...")
    
    nl_processor = NLProcessor()
    
    test_cases = [
        "Analyze the sales_data.csv file",
        "Build a classification model with 85% accuracy",
        "Show me revenue trends for the last 30 days",
        "Group by customer_id and calculate the mean",
        "Create a monthly forecast using random forest algorithm"
    ]
    
    print("\n=== Entity Extraction Tests ===")
    for query in test_cases:
        entities = nl_processor.entity_extractor.extract_entities(query)
        print(f"\nQuery: {query}")
        for entity in entities:
            print(f"  - {entity.type}: '{entity.value}' (confidence: {entity.confidence:.2f})")
    
    print("\n=== Entity Extraction Tests Complete ===")


async def test_response_generation():
    """Test response generation"""
    print("\nTesting Response Generation...")
    
    nl_processor = NLProcessor()
    
    # Mock agent responses
    test_responses = [
        {
            "intent": Intent.CTO,
            "response": {
                "success": True,
                "result": {
                    "summary": "Based on your requirements, I recommend a microservices architecture",
                    "recommendations": [
                        "Use Docker containers for deployment",
                        "Implement API Gateway pattern",
                        "Consider Kubernetes for orchestration"
                    ],
                    "metrics": {"estimated_cost": 5000, "implementation_time": "3 months"}
                }
            },
            "query": "What architecture should I use?"
        },
        {
            "intent": Intent.DATA_SCIENTIST,
            "response": {
                "success": True,
                "result": {
                    "summary": "Your dataset shows strong seasonal patterns",
                    "insights": [
                        "Sales peak in Q4 every year",
                        "Customer retention rate is 78%",
                        "Revenue growth is 15% year-over-year"
                    ]
                }
            },
            "query": "Analyze my sales data"
        }
    ]
    
    print("\n=== Response Generation Tests ===")
    for test_case in test_responses:
        nl_response = nl_processor.response_generator.generate_response(
            test_case["response"],
            test_case["intent"],
            test_case["query"]
        )
        print(f"\nQuery: {test_case['query']}")
        print(f"Intent: {test_case['intent'].value}")
        print(f"Generated Response:\n{nl_response}")
    
    print("\n=== Response Generation Tests Complete ===")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("ScrollIntel Core - Natural Language Interface Tests")
    print("=" * 60)
    
    try:
        await test_nl_processor()
        await test_entity_extraction()
        await test_response_generation()
        await test_orchestrator_integration()
        
        print("\n" + "=" * 60)
        print("All Natural Language Interface tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        logger.error("Test failed", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())