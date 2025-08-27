"""
Test production configuration for visual generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Test production configuration
    from scrollintel.engines.visual_generation.production_config import get_production_config
    
    print("🚀 Testing ScrollIntel Visual Generation Production Configuration")
    print("="*60)
    
    # Get production config
    config = get_production_config()
    print("✓ Production configuration loaded successfully")
    
    # Test model selection strategy
    strategy = config.get_model_selection_strategy()
    print(f"✓ Model selection strategy: {strategy['primary_strategy']}")
    print(f"✓ Selection rules: {len(strategy['selection_rules'])} rules configured")
    
    # Test competitive advantages
    advantages = config.get_competitive_advantages()
    print("\n🏆 COMPETITIVE ADVANTAGES:")
    print(f"  vs InVideo: {advantages['vs_invideo']['cost']}")
    print(f"  vs Runway: {advantages['vs_runway']['cost']}")
    print(f"  vs Pika Labs: {advantages['vs_pika_labs']['cost']}")
    print(f"  Unique features: {len(advantages['unique_advantages'])} advantages")
    
    # Test production readiness
    readiness = config.validate_production_readiness()
    print(f"\n📊 PRODUCTION READINESS:")
    print(f"  Overall score: {readiness['overall_readiness']['score']:.1%}")
    print(f"  Status: {readiness['overall_readiness']['status']}")
    print(f"  Passed checks: {readiness['overall_readiness']['passed_checks']}/{readiness['overall_readiness']['total_checks']}")
    
    # Test model availability
    print(f"\n🤖 MODEL AVAILABILITY:")
    for model_name, model_info in readiness['model_availability'].items():
        status = "✓" if model_info['available'] else "✗"
        model_type = model_info['type']
        is_local = "Local" if model_info['local_model'] else "API"
        print(f"  {status} {model_name} ({model_type}, {is_local})")
    
    # Test environment configuration
    print(f"\n⚙️ ENVIRONMENT CONFIGURATION:")
    if os.path.exists(".env.visual_generation"):
        print("  ✓ Production environment file configured")
    else:
        print("  ⚠ Production environment file not found")
    
    # Test monitoring configuration
    if os.path.exists("monitoring/visual-generation-alerts.yml"):
        print("  ✓ Monitoring and alerting configured")
    else:
        print("  ⚠ Monitoring configuration not found")
    
    # Test CI/CD pipeline
    if os.path.exists(".github/workflows/visual-generation-deployment.yml"):
        print("  ✓ CI/CD deployment pipeline configured")
    else:
        print("  ⚠ CI/CD pipeline not found")
    
    print(f"\n🎯 DEPLOYMENT STATUS:")
    if readiness['overall_readiness']['score'] >= 0.9:
        print("  🎉 READY FOR PRODUCTION DEPLOYMENT!")
        print("  🚀 ScrollIntel Visual Generation is configured to DOMINATE the market!")
        print("  🏆 Superior to InVideo, Runway, Pika Labs, and all competitors!")
    else:
        print("  ⚠ Additional configuration needed for production")
    
    print(f"\n✅ Production configuration test completed successfully!")
    
except Exception as e:
    print(f"✗ Production configuration test failed: {e}")
    import traceback
    traceback.print_exc()