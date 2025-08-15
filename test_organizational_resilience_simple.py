#!/usr/bin/env python3
"""
Simple test for Organizational Resilience Engine
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.engines.organizational_resilience_engine import OrganizationalResilienceEngine
from scrollintel.models.organizational_resilience_models import ResilienceCategory


async def test_resilience_engine():
    """Test the organizational resilience engine"""
    print("Testing Organizational Resilience Engine...")
    
    # Create engine instance
    engine = OrganizationalResilienceEngine()
    print("✅ Engine created successfully")
    
    # Test resilience assessment
    try:
        assessment = await engine.assess_organizational_resilience(
            organization_id="test_org_001"
        )
        print(f"✅ Assessment completed: {assessment.overall_resilience_level.value}")
        print(f"   Categories assessed: {len(assessment.category_scores)}")
        print(f"   Strengths: {len(assessment.strengths)}")
        print(f"   Vulnerabilities: {len(assessment.vulnerabilities)}")
        
        # Test strategy development
        strategy = await engine.develop_resilience_strategy(assessment)
        print(f"✅ Strategy developed: {len(strategy.initiatives)} initiatives")
        
        # Test monitoring
        monitoring_data = await engine.monitor_resilience_continuously("test_org_001")
        print(f"✅ Monitoring completed: {len(monitoring_data.metric_values)} metrics")
        
        # Test continuous improvement
        improvements = await engine.implement_continuous_improvement(
            "test_org_001", [monitoring_data]
        )
        print(f"✅ Improvements generated: {len(improvements)} recommendations")
        
        # Test report generation
        report = await engine.generate_resilience_report(
            "test_org_001", assessment, [monitoring_data], improvements
        )
        print(f"✅ Report generated: {report.overall_resilience_score:.2f} score")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_resilience_engine())
    sys.exit(0 if success else 1)