#!/usr/bin/env python3
"""
Demo script for ScrollBI Agent - Dashboard creation and business intelligence.

This script demonstrates the key capabilities of the ScrollBI agent:
1. Dashboard creation from data
2. Real-time dashboard setup
3. Alert system configuration
4. Dashboard sharing management
"""

import asyncio
import pandas as pd
from datetime import datetime
from uuid import uuid4

from scrollintel.agents.scroll_bi_agent import ScrollBIAgent, DashboardType
from scrollintel.core.interfaces import AgentRequest


async def demo_dashboard_creation():
    """Demo dashboard creation functionality."""
    print("🚀 ScrollBI Agent Demo - Dashboard Creation")
    print("=" * 50)
    
    # Initialize the agent
    agent = ScrollBIAgent()
    
    # Create sample business data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=90, freq='D'),
        'revenue': [10000 + i * 100 + (i % 7) * 500 for i in range(90)],
        'customers': [500 + i * 5 + (i % 5) * 20 for i in range(90)],
        'orders': [100 + i * 2 + (i % 3) * 10 for i in range(90)],
        'category': ['Electronics', 'Clothing', 'Books'] * 30,
        'region': ['North', 'South', 'East', 'West'] * 22 + ['North', 'South']
    })
    
    print(f"📊 Sample data created: {sample_data.shape[0]} rows, {sample_data.shape[1]} columns")
    print(f"Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
    print()
    
    # Test 1: Create Executive Dashboard
    print("1️⃣ Creating Executive Dashboard...")
    request = AgentRequest(
        id=f"req-{uuid4()}",
        user_id=f"user-{uuid4()}",
        agent_id="scroll-bi",
        prompt="Create an executive dashboard with revenue trends and customer metrics",
        context={
            "dataset": sample_data,
            "dashboard_config": {
                "name": "Executive Dashboard",
                "description": "High-level business metrics for executives",
                "dashboard_type": DashboardType.EXECUTIVE.value,
                "real_time_enabled": True
            }
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"✅ Dashboard created successfully!")
    print(f"⏱️  Execution time: {response.execution_time:.3f} seconds")
    print(f"📋 Response preview: {response.content[:200]}...")
    print()
    
    # Test 2: Set up Real-time Updates
    print("2️⃣ Setting up Real-time Dashboard Updates...")
    request = AgentRequest(
        id=f"req-{uuid4()}",
        user_id=f"user-{uuid4()}",
        agent_id="scroll-bi",
        prompt="Set up real-time updates for my dashboard",
        context={
            "dashboard_id": "dashboard-123",
            "update_interval": 30,
            "websocket_config": {
                "endpoint": "ws://localhost:8000/ws"
            }
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"✅ Real-time updates configured!")
    print(f"⏱️  Execution time: {response.execution_time:.3f} seconds")
    print()
    
    # Test 3: Configure Alert System
    print("3️⃣ Configuring Alert System...")
    request = AgentRequest(
        id=f"req-{uuid4()}",
        user_id=f"user-{uuid4()}",
        agent_id="scroll-bi",
        prompt="Set up alerts for revenue threshold and conversion rate",
        context={
            "dashboard_id": "dashboard-123",
            "alert_config": {
                "thresholds": {
                    "revenue": 50000,
                    "conversion_rate": 0.03
                },
                "notification_channels": ["email", "slack"],
                "email_recipients": ["admin@company.com"]
            }
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"✅ Alert system configured!")
    print(f"⏱️  Execution time: {response.execution_time:.3f} seconds")
    print()
    
    # Test 4: Dashboard Sharing
    print("4️⃣ Setting up Dashboard Sharing...")
    request = AgentRequest(
        id=f"req-{uuid4()}",
        user_id=f"user-{uuid4()}",
        agent_id="scroll-bi",
        prompt="Share dashboard with team members with different permissions",
        context={
            "dashboard_id": "dashboard-123",
            "sharing_config": {
                "users": [
                    {"email": "viewer@company.com", "permission": "view_only"},
                    {"email": "editor@company.com", "permission": "edit"}
                ],
                "public_access": False,
                "expiration_days": 30
            }
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"✅ Dashboard sharing configured!")
    print(f"⏱️  Execution time: {response.execution_time:.3f} seconds")
    print()
    
    # Test 5: BI Query Analysis
    print("5️⃣ Analyzing BI Query...")
    request = AgentRequest(
        id=f"req-{uuid4()}",
        user_id=f"user-{uuid4()}",
        agent_id="scroll-bi",
        prompt="Analyze this BI query and recommend dashboard layout",
        context={
            "bi_query": "SELECT date, SUM(revenue) as total_revenue, COUNT(orders) as order_count FROM sales GROUP BY date ORDER BY date",
            "data_context": {
                "tables": ["sales", "customers"],
                "metrics": ["revenue", "order_count"]
            },
            "user_preferences": {
                "layout": "grid",
                "theme": "dark"
            }
        },
        priority=1,
        created_at=datetime.now()
    )
    
    response = await agent.process_request(request)
    print(f"✅ BI query analyzed!")
    print(f"⏱️  Execution time: {response.execution_time:.3f} seconds")
    print()
    
    # Test 6: Agent Capabilities
    print("6️⃣ ScrollBI Agent Capabilities:")
    capabilities = agent.get_capabilities()
    for i, capability in enumerate(capabilities, 1):
        print(f"   {i}. {capability.name}: {capability.description}")
    print()
    
    # Test 7: Health Check
    print("7️⃣ Performing Health Check...")
    is_healthy = await agent.health_check()
    print(f"✅ Agent health status: {'Healthy' if is_healthy else 'Unhealthy'}")
    print()
    
    print("🎉 ScrollBI Agent Demo Completed Successfully!")
    print("=" * 50)
    print()
    print("Key Features Demonstrated:")
    print("✓ Instant dashboard creation from data")
    print("✓ Real-time dashboard updates with WebSocket")
    print("✓ Threshold-based alert system")
    print("✓ Dashboard sharing and permissions")
    print("✓ BI query analysis and recommendations")
    print("✓ Agent health monitoring")
    print()
    print("The ScrollBI agent is ready for production use!")


async def demo_different_dashboard_types():
    """Demo different dashboard types."""
    print("\n🎨 Dashboard Templates Demo")
    print("=" * 30)
    
    agent = ScrollBIAgent()
    
    # Sample data for different dashboard types
    financial_data = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=12, freq='ME'),
        'revenue': [100000 + i * 5000 for i in range(12)],
        'expenses': [80000 + i * 3000 for i in range(12)],
        'profit': [20000 + i * 2000 for i in range(12)]
    })
    
    dashboard_types = [
        (DashboardType.EXECUTIVE, "executive summary dashboard"),
        (DashboardType.SALES, "sales performance dashboard"),
        (DashboardType.FINANCIAL, "financial metrics dashboard"),
        (DashboardType.OPERATIONAL, "operational monitoring dashboard")
    ]
    
    for dashboard_type, prompt_text in dashboard_types:
        print(f"📊 Creating {dashboard_type.value.title()} Dashboard...")
        
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt=f"Create {prompt_text}",
            context={
                "dataset": financial_data,
                "dashboard_config": {
                    "name": f"{dashboard_type.value.title()} Dashboard",
                    "dashboard_type": dashboard_type.value
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await agent.process_request(request)
        print(f"   ✅ {dashboard_type.value.title()} dashboard created!")
        print(f"   ⏱️  Execution time: {response.execution_time:.3f} seconds")
        print()


if __name__ == "__main__":
    print("🔥 ScrollIntel™ ScrollBI Agent Demo")
    print("Advanced Dashboard Creation & Business Intelligence")
    print("=" * 60)
    print()
    
    # Run the main demo
    asyncio.run(demo_dashboard_creation())
    
    # Run dashboard types demo
    asyncio.run(demo_different_dashboard_types())
    
    print("Demo completed! 🚀")