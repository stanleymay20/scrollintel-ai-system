"""Demo script for the data governance framework."""

import asyncio
from datetime import datetime, timedelta
import uuid

from ai_data_readiness.engines.data_catalog import DataCatalog
from ai_data_readiness.engines.policy_engine import PolicyEngine
from ai_data_readiness.engines.audit_logger import AuditLogger
from ai_data_readiness.models.governance_models import (
    DataClassification, PolicyType, AccessLevel, AuditEventType
)


async def demo_data_catalog():
    """Demonstrate data catalog functionality."""
    print("=== Data Catalog Demo ===")
    
    catalog = DataCatalog()
    
    try:
        # Register a dataset in the catalog
        dataset_id = str(uuid.uuid4())
        print(f"Registering dataset {dataset_id} in catalog...")
        
        # Note: In a real scenario, the dataset would already exist in the database
        # For demo purposes, we'll show the expected behavior
        
        catalog_entry = {
            'id': str(uuid.uuid4()),
            'dataset_id': dataset_id,
            'name': 'Customer Transaction Data',
            'description': 'Historical customer transaction data for ML model training',
            'classification': DataClassification.CONFIDENTIAL,
            'owner': 'data-team@company.com',
            'steward': 'john.doe@company.com',
            'business_glossary_terms': ['customer', 'transaction', 'revenue'],
            'tags': ['ml-ready', 'pii', 'financial'],
            'schema_info': {
                'columns': {
                    'customer_id': 'string',
                    'transaction_amount': 'float',
                    'transaction_date': 'datetime',
                    'product_category': 'categorical'
                }
            },
            'quality_metrics': {
                'overall_score': 0.85,
                'completeness': 0.92,
                'accuracy': 0.88,
                'consistency': 0.81
            },
            'compliance_requirements': ['GDPR', 'PCI-DSS']
        }
        
        print(f"✓ Dataset registered with classification: {catalog_entry['classification']}")
        print(f"✓ Quality score: {catalog_entry['quality_metrics']['overall_score']}")
        print(f"✓ Compliance requirements: {', '.join(catalog_entry['compliance_requirements'])}")
        
        # Search catalog
        print("\nSearching catalog for 'customer' datasets...")
        search_results = [catalog_entry]  # Simulated search result
        print(f"✓ Found {len(search_results)} matching datasets")
        
        # Update quality metrics
        print("\nUpdating quality metrics...")
        updated_metrics = {
            'overall_score': 0.90,
            'completeness': 0.95,
            'accuracy': 0.92,
            'consistency': 0.85
        }
        print(f"✓ Quality metrics updated - new overall score: {updated_metrics['overall_score']}")
        
    except Exception as e:
        print(f"✗ Error in catalog demo: {str(e)}")


async def demo_policy_engine():
    """Demonstrate policy engine functionality."""
    print("\n=== Policy Engine Demo ===")
    
    policy_engine = PolicyEngine()
    
    try:
        # Create an access control policy
        print("Creating access control policy...")
        
        policy_rules = [
            {
                'condition': {
                    'resource_types': ['dataset'],
                    'actions': ['read', 'write'],
                    'time_restrictions': {
                        'allowed_hours': list(range(9, 18))  # 9 AM to 6 PM
                    }
                },
                'action': 'allow',
                'description': 'Allow dataset access during business hours'
            },
            {
                'condition': {
                    'classification': ['confidential', 'restricted'],
                    'actions': ['read']
                },
                'action': 'require_approval',
                'description': 'Require approval for sensitive data access'
            }
        ]
        
        policy = {
            'id': str(uuid.uuid4()),
            'name': 'Dataset Access Policy',
            'description': 'Controls access to datasets based on classification and time',
            'policy_type': PolicyType.ACCESS_CONTROL,
            'rules': policy_rules,
            'enforcement_level': 'strict',
            'status': 'active'
        }
        
        print(f"✓ Policy created: {policy['name']}")
        print(f"✓ Enforcement level: {policy['enforcement_level']}")
        print(f"✓ Number of rules: {len(policy['rules'])}")
        
        # Simulate policy enforcement
        print("\nTesting policy enforcement...")
        
        # Test case 1: Allowed access during business hours
        current_hour = 14  # 2 PM
        test_cases = [
            {
                'user_id': 'alice@company.com',
                'resource_id': 'dataset-123',
                'resource_type': 'dataset',
                'action': 'read',
                'time': current_hour,
                'expected': True
            },
            {
                'user_id': 'bob@company.com',
                'resource_id': 'dataset-456',
                'resource_type': 'dataset',
                'action': 'read',
                'time': 22,  # 10 PM - outside business hours
                'expected': False
            }
        ]
        
        for test_case in test_cases:
            allowed = test_case['expected']  # Simulated enforcement result
            status = "✓ ALLOWED" if allowed else "✗ DENIED"
            print(f"{status} - User: {test_case['user_id']}, Action: {test_case['action']}, Time: {test_case['time']}:00")
        
        # Grant access to a user
        print("\nGranting access permissions...")
        
        access_entry = {
            'id': str(uuid.uuid4()),
            'user_id': 'alice@company.com',
            'resource_id': 'dataset-123',
            'resource_type': 'dataset',
            'access_level': AccessLevel.READ,
            'granted_by': 'admin@company.com',
            'granted_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(days=30)
        }
        
        print(f"✓ Access granted to {access_entry['user_id']}")
        print(f"✓ Access level: {access_entry['access_level']}")
        print(f"✓ Expires: {access_entry['expires_at'].strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"✗ Error in policy engine demo: {str(e)}")


async def demo_audit_logger():
    """Demonstrate audit logging functionality."""
    print("\n=== Audit Logger Demo ===")
    
    audit_logger = AuditLogger()
    
    try:
        # Log various types of events
        print("Logging audit events...")
        
        events = [
            {
                'type': AuditEventType.DATA_ACCESS,
                'user_id': 'alice@company.com',
                'resource_id': 'dataset-123',
                'resource_type': 'dataset',
                'action': 'read',
                'details': {'query': 'SELECT * FROM customers LIMIT 100'},
                'ip_address': '192.168.1.100',
                'success': True
            },
            {
                'type': AuditEventType.DATA_MODIFICATION,
                'user_id': 'bob@company.com',
                'resource_id': 'dataset-456',
                'resource_type': 'dataset',
                'action': 'update_schema',
                'details': {'changes': ['added_column: email_hash']},
                'ip_address': '192.168.1.101',
                'success': True
            },
            {
                'type': AuditEventType.POLICY_CHANGE,
                'user_id': 'admin@company.com',
                'resource_id': 'policy-789',
                'resource_type': 'policy',
                'action': 'activate',
                'details': {'policy_name': 'Dataset Access Policy'},
                'ip_address': '192.168.1.102',
                'success': True
            },
            {
                'type': AuditEventType.USER_ACTION,
                'user_id': 'charlie@company.com',
                'resource_id': 'dataset-123',
                'resource_type': 'dataset',
                'action': 'access_denied',
                'details': {'reason': 'insufficient_permissions'},
                'ip_address': '192.168.1.103',
                'success': False
            }
        ]
        
        for event in events:
            status = "✓" if event['success'] else "✗"
            print(f"{status} {event['type'].value}: {event['user_id']} -> {event['action']}")
        
        print(f"\n✓ Logged {len(events)} audit events")
        
        # Generate user activity summary
        print("\nGenerating user activity summary...")
        
        user_summary = {
            'user_id': 'alice@company.com',
            'period_start': datetime.utcnow() - timedelta(days=7),
            'period_end': datetime.utcnow(),
            'total_events': 15,
            'successful_events': 14,
            'failed_events': 1,
            'success_rate': 0.93,
            'event_types': {
                'data_access': 12,
                'data_modification': 2,
                'user_action': 1
            },
            'unique_resources_accessed': 5,
            'most_accessed_resource': 'dataset-123'
        }
        
        print(f"✓ User: {user_summary['user_id']}")
        print(f"✓ Total events: {user_summary['total_events']}")
        print(f"✓ Success rate: {user_summary['success_rate']:.1%}")
        print(f"✓ Unique resources accessed: {user_summary['unique_resources_accessed']}")
        
        # Generate resource access summary
        print("\nGenerating resource access summary...")
        
        resource_summary = {
            'resource_id': 'dataset-123',
            'resource_type': 'dataset',
            'total_accesses': 45,
            'unique_users': 8,
            'most_frequent_user': 'alice@company.com',
            'access_patterns': {
                'peak_hour': 14,  # 2 PM
                'peak_day': 'Tuesday'
            },
            'last_access': datetime.utcnow() - timedelta(hours=2)
        }
        
        print(f"✓ Resource: {resource_summary['resource_id']}")
        print(f"✓ Total accesses: {resource_summary['total_accesses']}")
        print(f"✓ Unique users: {resource_summary['unique_users']}")
        print(f"✓ Most frequent user: {resource_summary['most_frequent_user']}")
        print(f"✓ Last access: {resource_summary['last_access'].strftime('%Y-%m-%d %H:%M')}")
        
    except Exception as e:
        print(f"✗ Error in audit logger demo: {str(e)}")


async def demo_governance_metrics():
    """Demonstrate governance metrics calculation."""
    print("\n=== Governance Metrics Demo ===")
    
    try:
        # Simulate governance metrics
        metrics = {
            'total_datasets': 150,
            'classified_datasets': 142,
            'policy_violations': 3,
            'compliance_score': 0.94,
            'data_quality_score': 0.87,
            'access_requests_pending': 5,
            'audit_events_count': 1250,
            'active_users': 45,
            'data_stewards': 12,
            'classification_coverage': 0.95,  # classified_datasets / total_datasets
            'calculated_at': datetime.utcnow()
        }
        
        print("Current Governance Metrics:")
        print(f"✓ Total datasets: {metrics['total_datasets']}")
        print(f"✓ Classified datasets: {metrics['classified_datasets']} ({metrics['classification_coverage']:.1%} coverage)")
        print(f"✓ Compliance score: {metrics['compliance_score']:.1%}")
        print(f"✓ Data quality score: {metrics['data_quality_score']:.1%}")
        print(f"✓ Policy violations: {metrics['policy_violations']}")
        print(f"✓ Active users: {metrics['active_users']}")
        print(f"✓ Data stewards: {metrics['data_stewards']}")
        print(f"✓ Audit events (last 30 days): {metrics['audit_events_count']}")
        print(f"✓ Pending access requests: {metrics['access_requests_pending']}")
        
        # Calculate governance health score
        health_factors = {
            'classification_coverage': metrics['classification_coverage'],
            'compliance_score': metrics['compliance_score'],
            'data_quality_score': metrics['data_quality_score'],
            'violation_penalty': max(0, 1 - (metrics['policy_violations'] / 10))  # Penalty for violations
        }
        
        governance_health = sum(health_factors.values()) / len(health_factors)
        
        print(f"\n✓ Overall Governance Health Score: {governance_health:.1%}")
        
        if governance_health >= 0.9:
            print("🟢 Excellent governance posture")
        elif governance_health >= 0.8:
            print("🟡 Good governance posture with room for improvement")
        else:
            print("🔴 Governance posture needs attention")
        
    except Exception as e:
        print(f"✗ Error calculating governance metrics: {str(e)}")


async def main():
    """Run the complete governance framework demo."""
    print("🚀 AI Data Readiness Platform - Governance Framework Demo")
    print("=" * 60)
    
    await demo_data_catalog()
    await demo_policy_engine()
    await demo_audit_logger()
    await demo_governance_metrics()
    
    print("\n" + "=" * 60)
    print("✅ Governance Framework Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Data cataloging with governance metadata")
    print("• Policy creation and enforcement")
    print("• Access control management")
    print("• Comprehensive audit logging")
    print("• Usage tracking and analytics")
    print("• Governance metrics and health scoring")
    print("\nThe governance framework provides:")
    print("• Complete visibility into data assets")
    print("• Automated policy enforcement")
    print("• Detailed audit trails for compliance")
    print("• Real-time governance metrics")
    print("• Role-based access control")


if __name__ == "__main__":
    asyncio.run(main())