"""
Simple test to verify optimization engine components work.
"""
from scrollintel.models.optimization_models import PerformanceMetrics, TestCase
from scrollintel.engines.optimization_engine import PerformanceEvaluator

def test_basic_functionality():
    """Test basic optimization functionality."""
    # Test performance evaluator
    evaluator = PerformanceEvaluator()
    
    test_cases = [
        TestCase(
            input_data={"query": "analyze the data"},
            expected_output="detailed analysis"
        )
    ]
    
    prompt = "Please analyze the following data carefully."
    metrics = evaluator.evaluate_prompt(prompt, test_cases)
    
    print(f"Metrics: accuracy={metrics.accuracy:.3f}, relevance={metrics.relevance:.3f}, efficiency={metrics.efficiency:.3f}")
    
    assert isinstance(metrics, PerformanceMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.relevance <= 1
    assert 0 <= metrics.efficiency <= 1
    
    print("✓ Performance evaluator works correctly")

if __name__ == "__main__":
    test_basic_functionality()
    print("All tests passed!")