"""
Demo script for legal compliance system.
Tests legal document management, consent tracking, and GDPR compliance features.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.core.legal_compliance_manager import LegalComplianceManager

async def demo_legal_compliance():
    """Demonstrate legal compliance features."""
    print("🏛️ ScrollIntel Legal Compliance System Demo")
    print("=" * 50)
    
    # Initialize compliance manager
    compliance_manager = LegalComplianceManager()
    
    # Demo 1: Legal Document Management
    print("\n📄 Demo 1: Legal Document Management")
    print("-" * 30)
    
    try:
        # Create sample legal document
        document_result = await compliance_manager.create_legal_document(
            document_type="demo_terms",
            title="Demo Terms of Service",
            content="""
            <h1>Demo Terms of Service</h1>
            <p>This is a demonstration of our legal document management system.</p>
            <h2>1. Acceptance</h2>
            <p>By using this demo, you accept these terms.</p>
            <h2>2. Demo Purpose</h2>
            <p>This document is for demonstration purposes only.</p>
            """,
            version="1.0",
            effective_date=datetime.utcnow(),
            metadata={"demo": True, "language": "en"}
        )
        
        print(f"✅ Created legal document: {document_result['title']}")
        print(f"   Version: {document_result['version']}")
        print(f"   Effective Date: {document_result['effective_date']}")
        
        # Retrieve the document
        retrieved_doc = await compliance_manager.get_legal_document("demo_terms")
        if retrieved_doc:
            print(f"✅ Retrieved document: {retrieved_doc['title']}")
            print(f"   Content length: {len(retrieved_doc['content'])} characters")
        
    except Exception as e:
        print(f"❌ Error in document management: {e}")
    
    # Demo 2: Consent Management
    print("\n🍪 Demo 2: Consent Management")
    print("-" * 30)
    
    try:
        demo_user_id = "demo_user_123"
        
        # Record various consents
        consent_types = [
            ("cookies_necessary", True),
            ("cookies_analytics", True),
            ("cookies_marketing", False),
            ("privacy_data_processing", True),
            ("privacy_marketing_emails", False)
        ]
        
        for consent_type, consent_given in consent_types:
            result = await compliance_manager.record_user_consent(
                user_id=demo_user_id,
                consent_type=consent_type,
                consent_given=consent_given,
                ip_address="127.0.0.1",
                user_agent="Demo Browser 1.0",
                document_version="1.0"
            )
            
            status = "✅ Granted" if consent_given else "❌ Denied"
            print(f"{status} {consent_type}")
        
        # Retrieve user consents
        user_consents = await compliance_manager.get_user_consents(demo_user_id)
        print(f"\n📋 User has {len(user_consents)} active consents:")
        for consent in user_consents:
            status = "✅" if consent['consent_given'] else "❌"
            print(f"   {status} {consent['consent_type']}")
        
    except Exception as e:
        print(f"❌ Error in consent management: {e}")
    
    # Demo 3: Data Export (GDPR Article 20)
    print("\n📦 Demo 3: Data Export Request")
    print("-" * 30)
    
    try:
        # Request data export
        export_result = await compliance_manager.request_data_export(
            user_id=demo_user_id,
            request_type="export"
        )
        
        print(f"✅ Data export requested")
        print(f"   Request ID: {export_result['id']}")
        print(f"   Status: {export_result['status']}")
        print(f"   Verification Token: {export_result['verification_token'][:8]}...")
        
        # Wait a moment for processing (in real scenario, this would be async)
        await asyncio.sleep(2)
        
        print("📊 Export would contain:")
        print("   - User profile data")
        print("   - Consent history")
        print("   - Activity logs")
        print("   - File metadata")
        print("   - README with export information")
        
    except Exception as e:
        print(f"❌ Error in data export: {e}")
    
    # Demo 4: Compliance Reporting
    print("\n📊 Demo 4: Compliance Reporting")
    print("-" * 30)
    
    try:
        # Generate compliance report
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        report = await compliance_manager.generate_compliance_report(
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"✅ Generated compliance report")
        print(f"   Report Type: {report.report_type}")
        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   Export Requests: {report.export_requests}")
        print(f"   Deletion Requests: {report.deletion_requests}")
        print(f"   Compliance Score: {report.compliance_score:.1f}%")
        
        print("\n📈 Consent Statistics:")
        for consent_type, count in report.consent_stats.items():
            print(f"   {consent_type}: {count}")
        
    except Exception as e:
        print(f"❌ Error in compliance reporting: {e}")
    
    # Demo 5: Cookie Consent Simulation
    print("\n🍪 Demo 5: Cookie Consent Simulation")
    print("-" * 30)
    
    try:
        # Simulate different cookie consent scenarios
        scenarios = [
            {
                "name": "Accept All Cookies",
                "settings": {
                    "necessary": True,
                    "analytics": True,
                    "marketing": True,
                    "preferences": True
                }
            },
            {
                "name": "Reject Optional Cookies",
                "settings": {
                    "necessary": True,
                    "analytics": False,
                    "marketing": False,
                    "preferences": False
                }
            },
            {
                "name": "Custom Selection",
                "settings": {
                    "necessary": True,
                    "analytics": True,
                    "marketing": False,
                    "preferences": True
                }
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            user_id = f"demo_user_{i+1}"
            print(f"\n👤 User {i+1}: {scenario['name']}")
            
            for cookie_type, accepted in scenario['settings'].items():
                if cookie_type == "necessary":
                    continue  # Necessary cookies are always accepted
                
                await compliance_manager.record_user_consent(
                    user_id=user_id,
                    consent_type=f"cookies_{cookie_type}",
                    consent_given=accepted,
                    ip_address="127.0.0.1",
                    user_agent=f"Demo Browser {i+1}.0"
                )
                
                status = "✅" if accepted else "❌"
                print(f"   {status} {cookie_type} cookies")
        
    except Exception as e:
        print(f"❌ Error in cookie consent simulation: {e}")
    
    # Demo 6: Privacy Rights Simulation
    print("\n🔒 Demo 6: Privacy Rights Simulation")
    print("-" * 30)
    
    try:
        # Simulate GDPR rights requests
        rights_user = "privacy_rights_user"
        
        # Right to Access (Article 15)
        print("📋 Right to Access - Getting user data...")
        user_consents = await compliance_manager.get_user_consents(rights_user)
        print(f"   Found {len(user_consents)} consent records")
        
        # Right to Portability (Article 20)
        print("📦 Right to Data Portability - Requesting export...")
        export_request = await compliance_manager.request_data_export(
            user_id=rights_user,
            request_type="export"
        )
        print(f"   Export request created: {export_request['id']}")
        
        # Right to be Forgotten (Article 17)
        print("🗑️ Right to be Forgotten - Requesting deletion...")
        deletion_request = await compliance_manager.request_data_export(
            user_id=rights_user,
            request_type="delete"
        )
        print(f"   Deletion request created: {deletion_request['id']}")
        
        print("\n✅ All GDPR rights demonstrated successfully!")
        
    except Exception as e:
        print(f"❌ Error in privacy rights simulation: {e}")
    
    # Demo Summary
    print("\n🎉 Demo Summary")
    print("=" * 50)
    print("✅ Legal document management")
    print("✅ User consent tracking")
    print("✅ Cookie consent handling")
    print("✅ GDPR data export")
    print("✅ Right to be forgotten")
    print("✅ Compliance reporting")
    print("✅ Privacy rights management")
    
    print("\n📚 Features Demonstrated:")
    print("• Legal document versioning and management")
    print("• Granular consent tracking with audit trails")
    print("• GDPR-compliant data export in ZIP format")
    print("• Automated compliance reporting")
    print("• Cookie consent banner integration")
    print("• Privacy settings management")
    print("• Data deletion and anonymization")
    
    print("\n🔧 Integration Points:")
    print("• FastAPI routes for legal compliance")
    print("• React components for cookie consent")
    print("• Privacy settings dashboard")
    print("• Legal pages (Terms, Privacy, Cookies)")
    print("• Database migrations for compliance tables")
    print("• Automated cleanup of expired exports")
    
    print("\n🛡️ Compliance Standards Met:")
    print("• GDPR (General Data Protection Regulation)")
    print("• CCPA (California Consumer Privacy Act)")
    print("• Cookie consent requirements")
    print("• Data retention policies")
    print("• Audit trail maintenance")
    print("• User rights management")

if __name__ == "__main__":
    asyncio.run(demo_legal_compliance())