"""
Demo script for the Production Upgrade Framework

This script demonstrates how to use the Production Upgrade Framework to assess
and upgrade ScrollIntel components to production standards.
"""

import asyncio
import sys
from pathlib import Path

# Add the scrollintel directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scrollintel.core.production_upgrade_framework import ProductionUpgradeEngine


async def demo_production_upgrade():
    """Demonstrate the production upgrade framework"""
    
    print("🚀 Production Upgrade Framework Demo")
    print("=" * 50)
    
    # Initialize the upgrade engine
    upgrade_engine = ProductionUpgradeEngine()
    
    # Example: Assess an existing component
    component_path = "scrollintel/engines/base_engine.py"
    
    if not Path(component_path).exists():
        print(f"❌ Component not found: {component_path}")
        return
    
    print(f"\n📊 Assessing component: {component_path}")
    
    try:
        # Get upgrade status
        status = await upgrade_engine.get_upgrade_status(component_path)
        
        print(f"   Production Readiness Score: {status['production_readiness_score']:.2f}/10")
        print(f"   Meets Production Standards: {'✅' if status['meets_standards'] else '❌'}")
        print(f"   Critical Issues: {status['critical_issues']}")
        print(f"   Recommendations: {status['recommendations']}")
        
        if not status['meets_standards']:
            print(f"\n🔧 Component needs upgrade. Starting upgrade process...")
            
            # Perform full upgrade
            result = await upgrade_engine.upgrade_component(component_path)
            
            print(f"\n📋 Upgrade Results:")
            print(f"   Success: {'✅' if result.success else '❌'}")
            print(f"   Completed Steps: {len(result.completed_steps)}")
            print(f"   Failed Steps: {len(result.failed_steps)}")
            print(f"   Execution Time: {result.execution_time}")
            
            if result.validation_results:
                passed = sum(1 for v in result.validation_results if v.passed)
                total = len(result.validation_results)
                print(f"   Validation: {passed}/{total} criteria passed")
        
        else:
            print(f"\n✅ Component already meets production standards!")
    
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_production_upgrade())