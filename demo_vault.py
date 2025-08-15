"""
Demo script for ScrollIntel Vault - Secure Insight Storage.
Demonstrates the key features of the vault system including encryption, 
access control, search, and audit functionality.
"""

import asyncio
import json
from datetime import datetime
from scrollintel.engines.vault_engine import ScrollVaultEngine, AccessLevel


async def demo_vault_functionality():
    """Demonstrate ScrollIntel Vault capabilities."""
    print("🔐 ScrollIntel Vault Demo - Secure Insight Storage")
    print("=" * 60)
    
    # Initialize vault engine
    print("\n1. Initializing Vault Engine...")
    vault = ScrollVaultEngine()
    await vault.initialize()
    
    # Set up user permissions for demo
    vault.user_permissions = {
        "data_scientist": AccessLevel.CONFIDENTIAL,
        "analyst": AccessLevel.INTERNAL,
        "viewer": AccessLevel.PUBLIC,
        "admin": AccessLevel.TOP_SECRET
    }
    
    print(f"✅ Vault initialized successfully")
    print(f"   - Encryption: {'✅ Available' if vault.cipher_suite else '❌ Mock mode'}")
    print(f"   - Semantic Search: {'✅ Available' if vault.embedding_model else '❌ Not available'}")
    
    # Demo 1: Store encrypted insights
    print("\n2. Storing Encrypted Insights...")
    
    insights_data = [
        {
            "title": "Customer Churn Analysis Results",
            "content": {
                "model_performance": {
                    "accuracy": 0.94,
                    "precision": 0.91,
                    "recall": 0.89,
                    "f1_score": 0.90
                },
                "key_findings": [
                    "High-value customers have 23% lower churn rate",
                    "Support ticket count strongly correlates with churn",
                    "Customers using mobile app are 15% less likely to churn"
                ],
                "recommendations": [
                    "Implement proactive support for high-ticket customers",
                    "Enhance mobile app engagement features",
                    "Create retention campaigns for at-risk segments"
                ]
            },
            "type": "analysis_result",
            "access_level": "confidential",
            "retention_policy": "long_term",
            "tags": ["churn", "ml", "customer_analytics"],
            "metadata": {
                "model_id": "churn_model_v3",
                "dataset": "customer_data_2024",
                "analyst": "sarah.johnson@company.com"
            }
        },
        {
            "title": "Q4 Sales Forecast Model",
            "content": {
                "forecast_results": {
                    "predicted_revenue": 2450000,
                    "confidence_interval": [2200000, 2700000],
                    "growth_rate": 0.12
                },
                "key_drivers": [
                    "Seasonal holiday boost: +18%",
                    "New product launch impact: +8%",
                    "Market expansion: +5%"
                ]
            },
            "type": "prediction",
            "access_level": "internal",
            "retention_policy": "medium_term",
            "tags": ["forecast", "sales", "revenue"],
            "metadata": {
                "model_type": "time_series",
                "forecast_horizon": "90_days",
                "created_by": "forecasting_team"
            }
        },
        {
            "title": "Data Quality Assessment Report",
            "content": {
                "overall_score": 0.87,
                "issues_found": {
                    "missing_values": 156,
                    "duplicates": 23,
                    "outliers": 45
                },
                "recommendations": [
                    "Implement data validation rules",
                    "Set up automated quality monitoring",
                    "Create data cleaning pipeline"
                ]
            },
            "type": "report",
            "access_level": "internal",
            "retention_policy": "short_term",
            "tags": ["data_quality", "assessment", "monitoring"],
            "metadata": {
                "dataset": "customer_transactions",
                "assessment_date": "2024-01-15"
            }
        }
    ]
    
    stored_insights = []
    for i, insight_data in enumerate(insights_data):
        result = await vault.process(
            input_data=insight_data,
            parameters={
                "operation": "store_insight",
                "user_id": "data_scientist",
                "organization_id": "demo_org",
                "ip_address": "192.168.1.100"
            }
        )
        stored_insights.append(result["insight_id"])
        print(f"   ✅ Stored insight {i+1}: {insight_data['title'][:40]}...")
        print(f"      ID: {result['insight_id']}")
        print(f"      Encrypted: {result['encrypted']}")
    
    # Demo 2: Retrieve and decrypt insights
    print("\n3. Retrieving and Decrypting Insights...")
    
    for i, insight_id in enumerate(stored_insights[:2]):  # Show first 2
        result = await vault.process(
            input_data=None,
            parameters={
                "operation": "retrieve_insight",
                "insight_id": insight_id,
                "user_id": "data_scientist",
                "ip_address": "192.168.1.100"
            }
        )
        
        insight = result["insight"]
        print(f"   📄 Insight {i+1}: {insight['title']}")
        print(f"      Type: {insight['type']}")
        print(f"      Access Level: {insight['access_level']}")
        print(f"      Access Count: {insight['access_count']}")
        print(f"      Content Keys: {list(insight['content'].keys())}")
    
    # Demo 3: Search functionality
    print("\n4. Searching Insights...")
    
    search_queries = [
        {"query": "churn analysis", "description": "Semantic search for churn"},
        {"query": "forecast", "description": "Search for forecasting insights"},
        {"query": "", "filters": {"tags": ["data_quality"]}, "description": "Filter by tags"}
    ]
    
    for search_data in search_queries:
        result = await vault.process(
            input_data=search_data,
            parameters={
                "operation": "search_insights",
                "user_id": "analyst",
                "ip_address": "192.168.1.101"
            }
        )
        
        print(f"   🔍 {search_data['description']}:")
        print(f"      Found {result['total_count']} insights")
        for insight in result["results"][:2]:  # Show first 2 results
            print(f"      - {insight['title'][:50]}...")
    
    # Demo 4: Access control
    print("\n5. Testing Access Control...")
    
    # Try to access confidential insight with low permissions
    confidential_insight_id = stored_insights[0]  # First insight is confidential
    
    try:
        await vault.process(
            input_data=None,
            parameters={
                "operation": "retrieve_insight",
                "insight_id": confidential_insight_id,
                "user_id": "viewer",  # Low permissions
                "ip_address": "192.168.1.102"
            }
        )
        print("   ❌ Access control failed - viewer should not access confidential data")
    except PermissionError:
        print("   ✅ Access control working - viewer blocked from confidential data")
    
    # Admin can access everything
    result = await vault.process(
        input_data=None,
        parameters={
            "operation": "retrieve_insight",
            "insight_id": confidential_insight_id,
            "user_id": "admin",
            "ip_address": "192.168.1.103"
        }
    )
    print("   ✅ Admin successfully accessed confidential insight")
    
    # Demo 5: Version control (update insight)
    print("\n6. Testing Version Control...")
    
    update_data = {
        "title": "Customer Churn Analysis Results - Updated",
        "content": {
            "model_performance": {
                "accuracy": 0.96,  # Improved accuracy
                "precision": 0.93,
                "recall": 0.91,
                "f1_score": 0.92
            },
            "update_notes": "Model retrained with additional features"
        },
        "tags": ["churn", "ml", "customer_analytics", "updated"]
    }
    
    result = await vault.process(
        input_data=update_data,
        parameters={
            "operation": "update_insight",
            "insight_id": confidential_insight_id,
            "user_id": "data_scientist",
            "ip_address": "192.168.1.100"
        }
    )
    
    print(f"   ✅ Created new version: {result['version']}")
    print(f"      New ID: {result['insight_id']}")
    print(f"      Parent ID: {result['parent_id']}")
    
    # Get version history
    history_result = await vault.process(
        input_data=None,
        parameters={
            "operation": "get_insight_history",
            "insight_id": confidential_insight_id,
            "user_id": "data_scientist"
        }
    )
    
    print(f"   📚 Version History ({history_result['total_versions']} versions):")
    for version in history_result["versions"]:
        print(f"      - Version {version['version']}: {version['title'][:40]}...")
    
    # Demo 6: Audit trail
    print("\n7. Reviewing Audit Trail...")
    
    audit_result = await vault.process(
        input_data=None,
        parameters={
            "operation": "audit_access",
            "user_id": "admin",  # Admin can see all logs
            "limit": 10,
            "offset": 0
        }
    )
    
    print(f"   📋 Found {audit_result['total_count']} audit log entries:")
    for log in audit_result["audit_logs"][:5]:  # Show first 5
        timestamp = datetime.fromisoformat(log["timestamp"]).strftime("%H:%M:%S")
        print(f"      - {timestamp}: {log['user_id']} {log['action']} insight {log['insight_id'][:8]}...")
    
    # Demo 7: Cleanup expired insights
    print("\n8. Testing Cleanup Functionality...")
    
    cleanup_result = await vault.process(
        input_data=None,
        parameters={
            "operation": "cleanup_expired",
            "user_id": "system"
        }
    )
    
    print(f"   🧹 Cleanup completed:")
    print(f"      Cleaned up: {cleanup_result['cleaned_up_count']} expired insights")
    
    # Demo 8: Vault statistics
    print("\n9. Vault Statistics...")
    
    status = vault.get_status()
    print(f"   📊 Vault Status:")
    print(f"      Engine ID: {status['engine_id']}")
    print(f"      Status: {status['status']}")
    print(f"      Stored Insights: {status['stored_insights']}")
    print(f"      Audit Logs: {status['audit_logs']}")
    print(f"      Cached Embeddings: {status['cached_embeddings']}")
    print(f"      Healthy: {status['healthy']}")
    
    print("\n✅ ScrollIntel Vault Demo Completed Successfully!")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("  🔐 End-to-end encryption of sensitive insights")
    print("  🛡️  Role-based access control with audit trails")
    print("  🔍 Semantic search and filtering capabilities")
    print("  📚 Version control with complete history tracking")
    print("  🧹 Automated cleanup of expired insights")
    print("  📊 Comprehensive monitoring and statistics")


if __name__ == "__main__":
    asyncio.run(demo_vault_functionality())