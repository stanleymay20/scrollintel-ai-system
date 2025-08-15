"""
Demo script for Natural Language Interface
Shows interactive conversation with ScrollIntel Core agents
"""
import asyncio
import logging
from typing import Dict, Any

from scrollintel_core.agents.orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for demo


async def interactive_demo():
    """Interactive demo of NL interface"""
    print("=" * 60)
    print("ScrollIntel Core - Natural Language Interface Demo")
    print("=" * 60)
    print("This demo shows how the NL interface processes queries and routes them to agents.")
    print("Type 'quit' to exit, 'help' for examples, or ask any question.\n")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    
    session_id = "demo_session"
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'help':
                show_help()
                continue
            
            if not user_input:
                continue
            
            print("\n" + "-" * 40)
            
            try:
                # Parse the query first to show NL processing
                parsed_info = await orchestrator.parse_query(user_input, session_id)
                
                print(f"🧠 NL Analysis:")
                print(f"   Intent: {parsed_info['intent']} (confidence: {parsed_info['confidence']:.2f})")
                print(f"   Suggested Agent: {parsed_info['suggested_agent']}")
                
                if parsed_info['entities']:
                    print(f"   Entities: {[(e['type'], e['value']) for e in parsed_info['entities']]}")
                
                if parsed_info['context_needed']:
                    print(f"   Context Needed: {parsed_info['context_needed']}")
                
                # Process the conversational request
                response = await orchestrator.process_conversational_request(
                    user_input, session_id
                )
                
                print(f"\n🤖 {response['agent']}: {response.get('nl_response', 'Processing complete.')}")
                
                if not response['success'] and response.get('error'):
                    print(f"   ⚠️  Error: {response['error']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40 + "\n")
    
    finally:
        await orchestrator.shutdown()
        print("\nDemo ended. Thanks for trying ScrollIntel Core!")


def show_help():
    """Show example queries"""
    print("\n📚 Example queries you can try:")
    print("\n🏗️  CTO Agent:")
    print("   - What's the best technology stack for a startup?")
    print("   - How should I scale my application architecture?")
    print("   - What infrastructure do I need for 1M users?")
    
    print("\n📊 Data Scientist Agent:")
    print("   - Analyze my sales data for trends")
    print("   - What patterns do you see in customer behavior?")
    print("   - Find correlations in my dataset")
    
    print("\n🤖 ML Engineer Agent:")
    print("   - Build a model to predict customer churn")
    print("   - Create a classification model for my data")
    print("   - Train a forecasting model")
    
    print("\n📈 BI Agent:")
    print("   - Create a dashboard for sales metrics")
    print("   - Show me KPIs for my business")
    print("   - Generate a monthly report")
    
    print("\n🧠 AI Engineer Agent:")
    print("   - What's the best AI strategy for my company?")
    print("   - How should I implement AI in my business?")
    print("   - Create an AI roadmap")
    
    print("\n❓ QA Agent:")
    print("   - How many customers are in my database?")
    print("   - What's the average order value?")
    print("   - Show me top selling products")
    
    print("\n🔮 Forecast Agent:")
    print("   - Predict sales for next quarter")
    print("   - Forecast customer growth")
    print("   - What will revenue be in 6 months?")
    print()


async def batch_demo():
    """Batch demo showing various NL capabilities"""
    print("=" * 60)
    print("ScrollIntel Core - NL Interface Batch Demo")
    print("=" * 60)
    
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    
    # Demo queries
    demo_queries = [
        "I need help choosing the right database for my application",
        "Analyze customer purchase patterns in my e-commerce data",
        "Build a machine learning model to predict which customers will churn",
        "Create a real-time dashboard showing key business metrics",
        "What's the best AI strategy for a retail company?",
        "How many orders were placed last month?",
        "Forecast our revenue growth for the next year"
    ]
    
    session_id = "batch_demo_session"
    
    try:
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 50)
            
            # Parse and show NL analysis
            parsed = await orchestrator.parse_query(query, session_id)
            print(f"Intent: {parsed['intent']} → Agent: {parsed['suggested_agent']}")
            print(f"Confidence: {parsed['confidence']:.2f}")
            
            if parsed['entities']:
                print(f"Entities: {[(e['type'], e['value']) for e in parsed['entities']]}")
            
            # Get suggestions
            suggestions = await orchestrator.get_nl_suggestions(query, session_id)
            if suggestions['suggestions']:
                print(f"Suggestions: {suggestions['suggestions'][0]}")
            
            print()
    
    finally:
        await orchestrator.shutdown()
    
    print("Batch demo complete!")


async def main():
    """Main demo function"""
    print("Choose demo mode:")
    print("1. Interactive demo (chat with agents)")
    print("2. Batch demo (see NL processing examples)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        await interactive_demo()
    elif choice == "2":
        await batch_demo()
    else:
        print("Invalid choice. Running interactive demo...")
        await interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())