"""
Demo script for workflow automation system.
"""
import asyncio
import json
from datetime import datetime

from scrollintel.engines.workflow_engine import WorkflowEngine
from scrollintel.models.workflow_models import (
    IntegrationType, TriggerType, ProcessingMode
)

async def demo_workflow_automation():
    """Demonstrate workflow automation capabilities."""
    print("🔄 ScrollIntel Workflow Automation Demo")
    print("=" * 50)
    
    engine = WorkflowEngine()
    
    # Demo 1: Create a simple custom workflow
    print("\n1. Creating Custom Workflow...")
    custom_workflow = {
        "name": "Data Processing Pipeline",
        "description": "Process incoming data and send notifications",
        "integration_type": IntegrationType.CUSTOM,
        "trigger_config": {
            "type": TriggerType.MANUAL,
            "config": {}
        },
        "steps": [
            {
                "name": "Validate Data",
                "type": "condition",
                "config": {
                    "condition": "data.get('amount', 0) > 0"
                }
            },
            {
                "name": "Transform Data",
                "type": "data_transformation",
                "config": {
                    "rules": [
                        {
                            "type": "map_field",
                            "source": "amount",
                            "target": "processed_amount"
                        }
                    ]
                }
            },
            {
                "name": "Send Notification",
                "type": "http_request",
                "config": {
                    "url": "https://httpbin.org/post",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json"
                    }
                }
            }
        ],
        "processing_mode": ProcessingMode.REAL_TIME
    }
    
    try:
        workflow_id = await engine.create_workflow(custom_workflow, "demo_user")
        print(f"✅ Created workflow: {workflow_id}")
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        return
    
    # Demo 2: Execute the workflow
    print("\n2. Executing Workflow...")
    test_data = {
        "amount": 100.50,
        "customer_id": "CUST123",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        execution_id = await engine.execute_workflow(workflow_id, test_data)
        print(f"✅ Started execution: {execution_id}")
    except Exception as e:
        print(f"❌ Error executing workflow: {e}")
    
    # Demo 3: Create Zapier integration workflow
    print("\n3. Creating Zapier Integration Workflow...")
    zapier_workflow = {
        "name": "Zapier Lead Processing",
        "description": "Process leads through Zapier automation",
        "integration_type": IntegrationType.ZAPIER,
        "trigger_config": {
            "type": TriggerType.WEBHOOK,
            "config": {
                "url": "https://hooks.zapier.com/hooks/catch/demo/webhook"
            }
        },
        "steps": [
            {
                "name": "Process Lead",
                "type": "zapier_action",
                "config": {
                    "webhook_url": "https://hooks.zapier.com/hooks/catch/demo/lead_process",
                    "action": "create_lead"
                }
            }
        ]
    }
    
    try:
        zapier_workflow_id = await engine.create_workflow(zapier_workflow, "demo_user")
        print(f"✅ Created Zapier workflow: {zapier_workflow_id}")
    except Exception as e:
        print(f"❌ Error creating Zapier workflow: {e}")
    
    # Demo 4: Create Power Automate integration workflow
    print("\n4. Creating Power Automate Integration Workflow...")
    power_automate_workflow = {
        "name": "Power Automate Document Processing",
        "description": "Process documents through Power Automate",
        "integration_type": IntegrationType.POWER_AUTOMATE,
        "trigger_config": {
            "type": TriggerType.EVENT,
            "config": {
                "event_type": "document_uploaded"
            }
        },
        "steps": [
            {
                "name": "Process Document",
                "type": "power_automate_flow",
                "config": {
                    "flow_url": "https://prod-123.westus.logic.azure.com/workflows/demo/triggers/manual/paths/invoke",
                    "flow_id": "demo-document-processing"
                }
            }
        ]
    }
    
    try:
        pa_workflow_id = await engine.create_workflow(power_automate_workflow, "demo_user")
        print(f"✅ Created Power Automate workflow: {pa_workflow_id}")
    except Exception as e:
        print(f"❌ Error creating Power Automate workflow: {e}")
    
    # Demo 5: Create Airflow integration workflow
    print("\n5. Creating Airflow Integration Workflow...")
    airflow_workflow = {
        "name": "Airflow Data Pipeline",
        "description": "Execute data pipeline through Airflow",
        "integration_type": IntegrationType.AIRFLOW,
        "trigger_config": {
            "type": TriggerType.SCHEDULE,
            "config": {
                "cron": "0 */6 * * *"  # Every 6 hours
            }
        },
        "steps": [
            {
                "name": "Run Data Pipeline",
                "type": "airflow_dag",
                "config": {
                    "base_url": "http://airflow.example.com",
                    "dag_id": "data_processing_pipeline",
                    "username": "admin",
                    "password": "password"
                }
            }
        ]
    }
    
    try:
        airflow_workflow_id = await engine.create_workflow(airflow_workflow, "demo_user")
        print(f"✅ Created Airflow workflow: {airflow_workflow_id}")
    except Exception as e:
        print(f"❌ Error creating Airflow workflow: {e}")
    
    # Demo 6: Demonstrate webhook management
    print("\n6. Webhook Management Demo...")
    webhook_config = {
        "url": "https://api.example.com/webhook/receive",
        "method": "POST",
        "headers": {
            "Authorization": "Bearer demo-token",
            "Content-Type": "application/json"
        },
        "secret": "webhook-secret-key"
    }
    
    try:
        webhook_id = await engine.webhook_manager.create_webhook(workflow_id, webhook_config)
        print(f"✅ Created webhook: {webhook_id}")
        
        # Simulate webhook callback
        callback_payload = {
            "event": "data_received",
            "data": {
                "amount": 250.75,
                "customer_id": "CUST456"
            }
        }
        
        callback_result = await engine.webhook_manager.handle_webhook_callback(webhook_id, callback_payload)
        print(f"✅ Webhook callback processed: {callback_result}")
        
    except Exception as e:
        print(f"❌ Error with webhook management: {e}")
    
    # Demo 7: Demonstrate retry mechanism
    print("\n7. Retry Mechanism Demo...")
    retry_count = 0
    
    async def failing_function():
        nonlocal retry_count
        retry_count += 1
        if retry_count < 3:
            raise Exception(f"Simulated failure #{retry_count}")
        return {"status": "success", "attempts": retry_count}
    
    try:
        result = await engine.retry_manager.execute_with_retry(failing_function)
        print(f"✅ Retry mechanism succeeded: {result}")
    except Exception as e:
        print(f"❌ Retry mechanism failed: {e}")
    
    # Demo 8: Workflow templates
    print("\n8. Workflow Templates Demo...")
    templates = engine.get_workflow_templates()
    print(f"📋 Available templates: {len(templates)}")
    
    for template in templates[:3]:  # Show first 3 templates
        print(f"  - {template['name']}: {template['description']}")
    
    # Demo 9: Batch processing workflow
    print("\n9. Batch Processing Demo...")
    batch_workflow = {
        "name": "Batch Data Processing",
        "description": "Process multiple records in batch",
        "integration_type": IntegrationType.CUSTOM,
        "trigger_config": {
            "type": TriggerType.SCHEDULE,
            "config": {
                "cron": "0 2 * * *"  # Daily at 2 AM
            }
        },
        "steps": [
            {
                "name": "Batch Transform",
                "type": "data_transformation",
                "config": {
                    "batch_size": 1000,
                    "rules": [
                        {
                            "type": "map_field",
                            "source": "raw_data",
                            "target": "processed_data"
                        }
                    ]
                }
            }
        ],
        "processing_mode": ProcessingMode.BATCH
    }
    
    try:
        batch_workflow_id = await engine.create_workflow(batch_workflow, "demo_user")
        print(f"✅ Created batch workflow: {batch_workflow_id}")
        
        # Execute with batch data
        batch_data = {
            "records": [
                {"raw_data": f"record_{i}"} for i in range(10)
            ]
        }
        
        batch_execution_id = await engine.execute_workflow(batch_workflow_id, batch_data)
        print(f"✅ Started batch execution: {batch_execution_id}")
        
    except Exception as e:
        print(f"❌ Error with batch processing: {e}")
    
    print("\n🎉 Workflow Automation Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Custom workflow creation and execution")
    print("✅ Zapier integration support")
    print("✅ Power Automate integration support")
    print("✅ Airflow integration support")
    print("✅ Webhook management and callbacks")
    print("✅ Retry mechanisms with exponential backoff")
    print("✅ Workflow templates and automation recipes")
    print("✅ Batch and real-time processing modes")
    print("✅ Error handling and recovery")

def demo_workflow_templates():
    """Demonstrate workflow template creation."""
    print("\n📋 Workflow Templates Demo")
    print("-" * 30)
    
    # Sample workflow templates
    templates = [
        {
            "name": "Lead Qualification Pipeline",
            "description": "Automatically qualify and route leads",
            "category": "sales",
            "integration_type": IntegrationType.ZAPIER,
            "template_config": {
                "trigger_config": {
                    "type": TriggerType.WEBHOOK,
                    "config": {"source": "web_form"}
                },
                "steps": [
                    {
                        "name": "Score Lead",
                        "type": "data_transformation",
                        "config": {
                            "rules": [
                                {"type": "calculate_score", "fields": ["company_size", "budget", "timeline"]}
                            ]
                        }
                    },
                    {
                        "name": "Route to Sales",
                        "type": "zapier_action",
                        "config": {"action": "create_salesforce_lead"}
                    }
                ]
            }
        },
        {
            "name": "Document Approval Workflow",
            "description": "Automate document review and approval",
            "category": "operations",
            "integration_type": IntegrationType.POWER_AUTOMATE,
            "template_config": {
                "trigger_config": {
                    "type": TriggerType.EVENT,
                    "config": {"event_type": "document_uploaded"}
                },
                "steps": [
                    {
                        "name": "Extract Metadata",
                        "type": "power_automate_flow",
                        "config": {"flow_id": "document_metadata_extraction"}
                    },
                    {
                        "name": "Route for Approval",
                        "type": "power_automate_flow",
                        "config": {"flow_id": "approval_routing"}
                    }
                ]
            }
        },
        {
            "name": "Data Quality Pipeline",
            "description": "Validate and clean incoming data",
            "category": "data",
            "integration_type": IntegrationType.AIRFLOW,
            "template_config": {
                "trigger_config": {
                    "type": TriggerType.SCHEDULE,
                    "config": {"cron": "0 1 * * *"}
                },
                "steps": [
                    {
                        "name": "Data Validation",
                        "type": "airflow_dag",
                        "config": {"dag_id": "data_quality_validation"}
                    },
                    {
                        "name": "Data Cleaning",
                        "type": "airflow_dag",
                        "config": {"dag_id": "data_cleaning_pipeline"}
                    }
                ]
            }
        }
    ]
    
    for template in templates:
        print(f"\n📄 {template['name']}")
        print(f"   Category: {template['category']}")
        print(f"   Integration: {template['integration_type']}")
        print(f"   Description: {template['description']}")
        print(f"   Steps: {len(template['template_config']['steps'])}")

if __name__ == "__main__":
    print("Starting Workflow Automation Demo...")
    asyncio.run(demo_workflow_automation())
    demo_workflow_templates()