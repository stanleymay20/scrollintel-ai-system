import asyncio
from scrollintel.engines.meta_learning_engine import MetaLearningEngine

async def test_meta_learning():
    """Simple test of meta-learning functionality."""
    engine = MetaLearningEngine()
    
    # Test rapid skill acquisition
    print("Testing rapid skill acquisition...")
    skill = await engine.rapid_skill_acquisition("test_skill", "analytical", 0.8)
    print(f"Skill acquired: {skill.skill_name}, mastery: {skill.mastery_level:.3f}")
    
    # Test transfer learning
    print("Testing transfer learning...")
    transfer_map = await engine.transfer_learning_across_domains("technical", "analytical")
    print(f"Transfer efficiency: {transfer_map.transfer_efficiency:.3f}")
    
    # Test self-improvement
    print("Testing self-improvement...")
    improvement_plan = await engine.self_improving_algorithms("reasoning_speed", 1.3)
    print(f"Improvement target: {improvement_plan.improvement_target}")
    print(f"Current -> Target: {improvement_plan.current_capability:.3f} -> {improvement_plan.target_capability:.3f}")
    
    # Test adaptation to new environments
    print("Testing environment adaptation...")
    env_desc = {"type": "complex", "complexity": 0.8, "volatility": 0.6}
    adaptation_result = await engine.adapt_to_new_environments(env_desc)
    print(f"Adaptation success: {adaptation_result['success']}")
    
    # Get learning statistics
    stats = engine.get_learning_statistics()
    print(f"Learning statistics: {stats}")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_meta_learning())