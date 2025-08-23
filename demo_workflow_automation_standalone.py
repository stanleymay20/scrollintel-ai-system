"""
Standalone demo for workflow automation system components.
"""
import asyncio
from unittest.mock import Mock

# Import the integration classes directly
from scrollintel.engines.workflow_engine import (
    ZapierIntegration, PowerAutomateIntegration, 
    AirflowIntegration, CustomIntegration, RetryManager
)

async def demo_integrations():
    """Demonstrate workflow integration capabilities."""
    print("🔄 ScrollIntel Workflow Automation Integration Demo")
    print("=" * 60)
    
    # Demo 1: Custom Integration
    print("\n1. Custom Integration Demo")
    print("-" * 30)
    
    custom_integration = CustomIntegration()
    
    # Test data transformation
    transform_config = {
        "type": "data_transformation",
        "config": {
            "rules": [
                {
                    "type": "map_field",
                    "source": "customer_name",
                    "target": "name"
                },
                {
                    "type": "map_field", 
                    "source": "order_amount",
                    "target": "amount"
                }
            ]
        }
    }
    
    input_data = {
        "customer_name": "John Doe",
        "order_amount": 150.75,
        "order_date": "2024-01-15"
    }
    
    try:
        result = await custom_integration.execute_step(transform_config, input_data)
        print(f"✅ Data transformation result: {result}")
    except Exception as e:
        print(f"❌ Data transformation error: {e}")
    
    # Test condition evaluation
    condition_config = {
        "type": "condition",
        "config": {
            "condition": "data.get('amount', 0) > 100"
        }
    }
    
    try:
        result = await custom_integration.execute_step(condition_config, {"amount": 150.75})
        print(f"✅ Condition evaluation (>100): {result['condition_met']}")
        
        result = await custom_integration.execute_step(condition_config, {"amount": 50.25})
        print(f"✅ Condition evaluation (≤100): {result['condition_met']}")
    except Exception as e:
        print(f"❌ Condition evaluation error: {e}")
    
    # Demo 2: Retry Manager
    print("\n2. Retry Manager Demo")
    print("-" * 30)
    
    retry_manager = RetryManager()
    
    # Simulate a function that fails twice then succeeds
    attempt_count = 0
    
    async def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        print(f"   Attempt #{attempt_count}")
        
        if attempt_count < 3:
            raise Exception(f"Simulated failure #{attempt_count}")
        return {"status": "success", "attempts": attempt_count}
    
    try:
        result = await retry_manager.execute_with_retry(flaky_function)
        print(f"✅ Retry manager succeeded: {result}")
    except Exception as e:
        print(f"❌ Retry manager failed: {e}")
    
    # Demo 3: Integration Types
    print("\n3. Integration Types Demo")
    print("-" * 30)
    
    integrations = {
        "Zapier": ZapierIntegration(),
        "Power Automate": PowerAutomateIntegration(),
        "Airflow": AirflowIntegration(),
        "Custom": CustomIntegration()
    }
    
    for name, integration in integrations.items():
        print(f"✅ {name} integration initialized")
    
    # Demo 4: Workflow Step Configurations
    print("\n4. Workflow Step Configuration Examples")
    print("-" * 30)
    
    step_examples = [
        {
            "name": "Zapier Lead Processing",
            "type": "zapier_action",
            "config": {
                "webhook_url": "https://hooks.zapier.com/hooks/catch/123/abc",
                "action": "create_lead"
            }
        },
        {
            "name": "Power Automate Document Review",
            "type": "power_automate_flow",
            "config": {
                "flow_url": "https://prod-123.westus.logic.azure.com/workflows/doc-review",
                "flow_id": "document-approval-flow"
            }
        },
        {
            "name": "Airflow Data Pipeline",
            "type": "airflow_dag",
            "config": {
                "base_url": "http://airflow.company.com",
                "dag_id": "data_processing_pipeline",
                "username": "admin"
            }
        },
        {
            "name": "HTTP API Call",
            "type": "http_request",
            "config": {
                "url": "https://api.example.com/process",
                "method": "POST",
                "headers": {"Content-Type": "application/json"}
            }
        },
        {
            "name": "Data Validation",
            "type": "condition",
            "config": {
                "condition": "data.get('email') and '@' in data['email']"
            }
        }
    ]
    
    for step in step_examples:
        print(f"📋 {step['name']}")
        print(f"   Type: {step['type']}")
        print(f"   Config keys: {list(step['config'].keys())}")
    
    # Demo 5: Workflow Templates
    print("\n5. Workflow Template Examples")
    print("-" * 30)
    
    templates = [
        {
            "name": "Customer Onboarding Pipeline",
            "description": "Automate new customer setup process",
            "integration": "Zapier + Custom",
            "steps": [
                "Validate customer data",
                "Create CRM record",
                "Send welcome email",
                "Schedule follow-up"
            ]
        },
        {
            "name": "Invoice Processing Workflow",
            "description": "Process and approve invoices automatically",
            "integration": "Power Automate + Custom",
            "steps": [
                "Extract invoice data",
                "Validate against PO",
                "Route for approval",
                "Update accounting system"
            ]
        },
        {
            "name": "Data Quality Monitoring",
            "description": "Monitor and clean data quality issues",
            "integration": "Airflow + Custom",
            "steps": [
                "Run data quality checks",
                "Identify anomalies",
                "Apply cleaning rules",
                "Generate quality report"
            ]
        }
    ]
    
    for template in templates:
        print(f"📄 {template['name']}")
        print(f"   Integration: {template['integration']}")
        print(f"   Description: {template['description']}")
        print(f"   Steps: {len(template['steps'])}")
        for i, step in enumerate(template['steps'], 1):
            print(f"     {i}. {step}")
        print()
    
    print("🎉 Workflow Automation Integration Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("✅ Custom data transformation and validation")
    print("✅ Retry mechanisms with exponential backoff")
    print("✅ Multiple integration types (Zapier, Power Automate, Airflow)")
    print("✅ Flexible step configuration system")
    print("✅ Workflow template examples")
    print("✅ Error handling and recovery patterns")

def demo_workflow_patterns():
    """Demonstrate common workflow patterns."""
    print("\n🔄 Common Workflow Patterns")
    print("=" * 40)
    
    patterns = [
        {
            "name": "Sequential Processing",
            "description": "Execute steps one after another",
            "example": "Data ingestion → Validation → Transformation → Storage"
        },
        {
            "name": "Conditional Branching", 
            "description": "Execute different paths based on conditions",
            "example": "Check order amount → If >$1000: Manager approval → Else: Auto-approve"
        },
        {
            "name": "Parallel Processing",
            "description": "Execute multiple steps simultaneously",
            "example": "Send email + Update CRM + Log event (all in parallel)"
        },
        {
            "name": "Error Handling",
            "description": "Handle failures gracefully with retries",
            "example": "API call → If fails: Retry 3x → If still fails: Alert admin"
        },
        {
            "name": "Webhook Triggered",
            "description": "Start workflow from external events",
            "example": "Form submission → Validate data → Create lead → Notify sales"
        },
        {
            "name": "Scheduled Batch",
            "description": "Process data in scheduled batches",
            "example": "Daily at 2 AM: Collect logs → Analyze → Generate report"
        }
    ]
    
    for pattern in patterns:
        print(f"\n📋 {pattern['name']}")
        print(f"   {pattern['description']}")
        print(f"   Example: {pattern['example']}")

if __name__ == "__main__":
    print("Starting Workflow Automation Integration Demo...")
    asyncio.run(demo_integrations())
    demo_workflow_patterns()