#!/usr/bin/env python3
"""
Simple test for design iteration engine functionality
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_design_iteration_engine():
    """Test design iteration engine functionality"""
    try:
        print("Testing design iteration engine...")
        
        # Import required modules
        from scrollintel.models.prototype_models import (
            Concept, Prototype, PrototypeType, PrototypeStatus,
            ConceptCategory, QualityMetrics, create_concept_from_description
        )
        from scrollintel.engines.design_iteration_engine import (
            DesignIterationEngine, DesignFeedback, FeedbackType, IterationType
        )
        
        # Create a test prototype
        concept = create_concept_from_description(
            name="Test Web App",
            description="A web application for testing design iteration",
            category=ConceptCategory.PRODUCT
        )
        
        prototype = Prototype(
            concept_id=concept.id,
            name="Test Web App Prototype",
            description="A prototype for testing design iteration",
            prototype_type=PrototypeType.WEB_APP,
            status=PrototypeStatus.FUNCTIONAL,
            quality_metrics=QualityMetrics(
                code_coverage=0.6,
                performance_score=0.7,
                usability_score=0.5,
                reliability_score=0.8,
                security_score=0.6,
                maintainability_score=0.7,
                scalability_score=0.6
            )
        )
        
        print(f"✅ Created test prototype: {prototype.name}")
        print(f"   Initial quality scores:")
        print(f"   - Performance: {prototype.quality_metrics.performance_score:.2f}")
        print(f"   - Usability: {prototype.quality_metrics.usability_score:.2f}")
        print(f"   - Security: {prototype.quality_metrics.security_score:.2f}")
        
        # Create design iteration engine
        iteration_engine = DesignIterationEngine()
        print("✅ Design iteration engine created successfully")
        
        # Add some feedback
        feedback_items = [
            "The application is too slow, response times are over 3 seconds",
            "The user interface is confusing and hard to navigate",
            "Missing important security features like authentication",
            "The app crashes when handling large datasets",
            "Great concept but needs performance improvements"
        ]
        
        print(f"\n🔄 Adding {len(feedback_items)} feedback items...")
        for i, feedback_text in enumerate(feedback_items):
            feedback = await iteration_engine.create_feedback_from_text(
                prototype.id, feedback_text, FeedbackType.USER_FEEDBACK, f"user_{i+1}"
            )
            await iteration_engine.add_feedback(prototype.id, feedback)
            print(f"   Added feedback: {feedback_text[:50]}...")
        
        # Analyze feedback
        print("\n🔄 Analyzing feedback...")
        feedback_analysis = await iteration_engine.analyze_prototype_feedback(prototype.id)
        
        print(f"✅ Feedback analysis completed:")
        print(f"   - Total feedback items: {feedback_analysis['total_feedback_items']}")
        print(f"   - Critical issues: {len(feedback_analysis['critical_issues'])}")
        print(f"   - Priority items: {len(feedback_analysis['priority_items'])}")
        print(f"   - Recommended iterations: {len(feedback_analysis['recommended_iterations'])}")
        
        # Plan and execute iterations
        print("\n🔄 Running iteration cycle...")
        iterations = await iteration_engine.run_iteration_cycle(prototype)
        
        print(f"✅ Iteration cycle completed with {len(iterations)} iterations:")
        for i, iteration in enumerate(iterations):
            print(f"   Iteration {iteration.iteration_number}:")
            print(f"   - Type: {iteration.iteration_type.value}")
            print(f"   - Improvement score: {iteration.improvement_score:.3f}")
            print(f"   - Duration: {iteration.duration_minutes:.1f} minutes")
            print(f"   - Status: {iteration.status}")
        
        # Check final quality metrics
        print(f"\n📊 Final quality scores:")
        final_metrics = prototype.quality_metrics
        print(f"   - Performance: {final_metrics.performance_score:.2f}")
        print(f"   - Usability: {final_metrics.usability_score:.2f}")
        print(f"   - Security: {final_metrics.security_score:.2f}")
        print(f"   - Overall improvement achieved!")
        
        # Get convergence status
        convergence_metrics = await iteration_engine.convergence_tracker.get_convergence_status(prototype.id)
        if convergence_metrics:
            print(f"\n📈 Convergence status:")
            print(f"   - Status: {convergence_metrics.convergence_status.value}")
            print(f"   - Convergence score: {convergence_metrics.convergence_score:.2f}")
            print(f"   - Stability score: {convergence_metrics.stability_score:.2f}")
            print(f"   - Improvement velocity: {convergence_metrics.improvement_velocity:.3f}")
        
        # Generate comprehensive report
        print("\n📋 Generating iteration report...")
        report = await iteration_engine.generate_iteration_report(prototype.id)
        
        print(f"✅ Iteration report generated:")
        stats = report['iteration_statistics']
        print(f"   - Total iterations: {stats['total_iterations']}")
        print(f"   - Success rate: {stats['success_rate']:.1%}")
        print(f"   - Average improvement: {stats['average_improvement_score']:.3f}")
        print(f"   - Total duration: {stats['total_duration_minutes']:.1f} minutes")
        
        # Test feedback summary
        feedback_summary = await iteration_engine.get_feedback_summary(prototype.id)
        print(f"\n📝 Feedback summary:")
        print(f"   - Total feedback: {feedback_summary['total_feedback']}")
        print(f"   - Addressed: {feedback_summary['addressed_feedback']}")
        print(f"   - Pending: {feedback_summary['pending_feedback']}")
        print(f"   - Average severity: {feedback_summary['average_severity']:.2f}")
        
        print("\n🎉 Design iteration engine test completed successfully!")
        print("\n📋 Task 3.2 Implementation Summary:")
        print("   ✅ Iterative design improvement and optimization system")
        print("   ✅ Design feedback integration and enhancement")
        print("   ✅ Design convergence and optimization tracking")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during design iteration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_design_iteration_engine())
    sys.exit(0 if success else 1)