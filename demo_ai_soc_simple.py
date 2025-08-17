"""
Simple demonstration of AI-Enhanced Security Operations Center
"""
import asyncio
import logging
from datetime import datetime

from security.ai_soc.ai_soc_orchestrator import AISOCOrchestrator
from security.ai_soc.ml_siem_engine import SecurityEvent, EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Simple AI SOC demonstration"""
    print("🚀 AI-Enhanced Security Operations Center - Simple Demo")
    print("=" * 55)
    
    # Initialize AI SOC
    print("\n🔧 Initializing AI SOC...")
    ai_soc = AISOCOrchestrator()
    
    try:
        await ai_soc.initialize()
        print("✅ AI SOC initialized successfully!")
        
        # Create a sample security event
        print("\n📊 Processing sample security event...")
        
        sample_event = SecurityEvent(
            event_id="demo_001",
            timestamp=datetime.now(),
            event_type=EventType.LOGIN_ATTEMPT,
            source_ip="192.168.1.100",
            user_id="demo_user",
            resource="login_portal",
            raw_data={
                "success": False,
                "attempts": 3,
                "user_agent": "Mozilla/5.0"
            },
            risk_score=0.6
        )
        
        # Process the event
        result = await ai_soc.process_security_event(sample_event)
        
        print(f"✅ Event processed successfully!")
        print(f"   • Event ID: {result['event_id']}")
        print(f"   • Actions taken: {len(result['actions_taken'])}")
        print(f"   • Alerts generated: {len(result.get('alerts', []))}")
        print(f"   • Correlations found: {len(result.get('correlations', []))}")
        print(f"   • Incidents created: {len(result.get('incidents', []))}")
        
        # Get performance metrics
        print("\n📈 Performance Metrics:")
        metrics = ai_soc.get_comprehensive_metrics()
        
        soc_metrics = metrics["soc_metrics"]
        print(f"   • Events processed: {soc_metrics['events_processed']}")
        print(f"   • Alerts generated: {soc_metrics['alerts_generated']}")
        print(f"   • False positive rate: {soc_metrics['false_positive_rate']:.1%}")
        
        # Get SOC dashboard
        print("\n🎛️ SOC Dashboard:")
        dashboard = await ai_soc.get_soc_dashboard()
        print(f"   • Overall risk score: {dashboard.overall_risk_score:.2f}")
        print(f"   • Active incidents: {dashboard.active_incidents}")
        print(f"   • System health: {dashboard.system_health['overall']}")
        
        print("\n✅ Demo completed successfully!")
        print("🎯 AI SOC is ready for enterprise security operations!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())