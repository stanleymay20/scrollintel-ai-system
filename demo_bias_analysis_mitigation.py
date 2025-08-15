"""Demo script for AI Data Readiness Platform Bias Analysis and Mitigation."""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.engines.bias_mitigation_engine import (
    BiasMitigationEngine, FairnessConstraint
)
from ai_data_readiness.models.base_models import BiasType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_biased_dataset():
    """Create a sample biased dataset for demonstration."""
    np.random.seed(42)
    n_samples = 2000
    
    # Create biased hiring dataset
    data = {
        'gender': np.random.choice(['male', 'female'], n_samples, p=[0.75, 0.25]),
        'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], 
                                n_samples, p=[0.65, 0.15, 0.15, 0.05]),
        'age': np.random.randint(22, 65, n_samples),
        'education': np.random.choice(['bachelor', 'master', 'phd'], 
                                    n_samples, p=[0.6, 0.3, 0.1]),
        'experience_years': np.random.randint(0, 20, n_samples),
        'salary_expectation': np.random.normal(75000, 25000, n_samples),
        'interview_score': np.random.normal(7.5, 1.5, n_samples)
    }
    
    # Introduce bias in hiring decisions
    # Males and whites have higher probability of being hired
    hire_prob = np.ones(n_samples) * 0.3  # Base probability
    
    # Gender bias
    male_mask = np.array(data['gender']) == 'male'
    hire_prob[male_mask] += 0.3
    
    # Race bias
    white_mask = np.array(data['race']) == 'white'
    hire_prob[white_mask] += 0.2
    
    # Add some legitimate factors
    high_score_mask = np.array(data['interview_score']) > 8.0
    hire_prob[high_score_mask] += 0.2
    
    high_exp_mask = np.array(data['experience_years']) > 10
    hire_prob[high_exp_mask] += 0.15
    
    # Ensure probabilities are valid
    hire_prob = np.clip(hire_prob, 0, 1)
    
    # Generate hiring decisions
    data['hired'] = np.random.binomial(1, hire_prob, n_samples)
    
    return pd.DataFrame(data)


def demonstrate_bias_detection():
    """Demonstrate bias detection capabilities."""
    print("=" * 60)
    print("AI DATA READINESS PLATFORM - BIAS ANALYSIS DEMO")
    print("=" * 60)
    
    # Create biased dataset
    print("\n1. Creating biased hiring dataset...")
    data = create_biased_dataset()
    print(f"Dataset created with {len(data)} samples")
    print(f"Columns: {list(data.columns)}")
    
    # Show basic statistics
    print("\n2. Dataset Overview:")
    print(f"Gender distribution: {data['gender'].value_counts().to_dict()}")
    print(f"Race distribution: {data['race'].value_counts().to_dict()}")
    print(f"Hiring rate by gender: {data.groupby('gender')['hired'].mean().to_dict()}")
    print(f"Hiring rate by race: {data.groupby('race')['hired'].mean().to_dict()}")
    
    # Initialize bias analysis engine
    print("\n3. Initializing Bias Analysis Engine...")
    bias_engine = BiasAnalysisEngine()
    
    # Detect bias
    print("\n4. Detecting bias in the dataset...")
    protected_attributes = ['gender', 'race']
    bias_report = bias_engine.detect_bias(
        dataset_id="hiring_dataset_demo",
        data=data,
        protected_attributes=protected_attributes,
        target_column='hired'
    )
    
    # Display results
    print(f"\nBias Analysis Results:")
    print(f"Protected attributes analyzed: {bias_report.protected_attributes}")
    print(f"Number of fairness violations detected: {len(bias_report.fairness_violations)}")
    
    print("\nBias Metrics:")
    for attr, metrics in bias_report.bias_metrics.items():
        print(f"\n{attr.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.3f}")
    
    print("\nFairness Violations:")
    for i, violation in enumerate(bias_report.fairness_violations, 1):
        print(f"\n{i}. {violation.bias_type.value.replace('_', ' ').title()}")
        print(f"   Attribute: {violation.protected_attribute}")
        print(f"   Severity: {violation.severity}")
        print(f"   Description: {violation.description}")
        print(f"   Affected groups: {violation.affected_groups}")
    
    return bias_report, data


def demonstrate_bias_mitigation(bias_report, data):
    """Demonstrate bias mitigation recommendations."""
    print("\n" + "=" * 60)
    print("BIAS MITIGATION RECOMMENDATIONS")
    print("=" * 60)
    
    # Initialize mitigation engine
    print("\n1. Initializing Bias Mitigation Engine...")
    mitigation_engine = BiasMitigationEngine()
    
    # Define fairness constraints
    print("\n2. Defining fairness constraints...")
    constraints = [
        FairnessConstraint(
            metric_name="demographic_parity",
            threshold=0.1,
            operator="less_than",
            protected_attribute="gender",
            priority="high"
        ),
        FairnessConstraint(
            metric_name="disparate_impact",
            threshold=0.8,
            operator="greater_than",
            protected_attribute="race",
            priority="high"
        )
    ]
    
    print("Fairness constraints defined:")
    for constraint in constraints:
        print(f"  - {constraint.metric_name} for {constraint.protected_attribute} "
              f"should be {constraint.operator} {constraint.threshold}")
    
    # Get comprehensive mitigation recommendations
    print("\n3. Generating mitigation recommendations...")
    recommendations = mitigation_engine.recommend_mitigation_approach(
        bias_report, data, constraints
    )
    
    # Display recommended strategies
    print(f"\nRecommended Mitigation Strategies ({len(recommendations['recommended_strategies'])}):")
    for i, strategy in enumerate(recommendations['recommended_strategies'][:5], 1):  # Show top 5
        print(f"\n{i}. {strategy.description}")
        print(f"   Type: {strategy.strategy_type}")
        print(f"   Expected Impact: {strategy.expected_impact:.1%}")
        print(f"   Complexity: {strategy.complexity}")
        print(f"   Key Steps:")
        for step in strategy.implementation_steps[:3]:  # Show first 3 steps
            print(f"     • {step}")
        if len(strategy.implementation_steps) > 3:
            print(f"     • ... and {len(strategy.implementation_steps) - 3} more steps")
    
    # Display implementation roadmap
    print(f"\nImplementation Roadmap:")
    for phase in recommendations['implementation_roadmap']:
        print(f"\nPhase {phase['phase']}: {phase['name']}")
        print(f"  Duration: {phase['duration']}")
        print(f"  Description: {phase['description']}")
        print(f"  Strategies: {len(phase['strategies'])}")
    
    # Display constraint validation
    print(f"\nConstraint Validation Results:")
    for constraint, satisfied in recommendations['constraint_validation'].items():
        status = "✓ PASSED" if satisfied else "✗ FAILED"
        print(f"  {constraint}: {status}")
    
    # Display timeline and resources
    print(f"\nProject Estimates:")
    print(f"  Estimated Timeline: {recommendations['estimated_timeline']}")
    print(f"  Team Size: {recommendations['resource_requirements']['team_size']}")
    print(f"  Skills Required: {', '.join(recommendations['resource_requirements']['skills_required'])}")
    print(f"  Budget Estimate: {recommendations['resource_requirements']['budget_estimate']}")
    
    return recommendations


def demonstrate_fairness_validation(data):
    """Demonstrate fairness constraint validation."""
    print("\n" + "=" * 60)
    print("FAIRNESS CONSTRAINT VALIDATION")
    print("=" * 60)
    
    mitigation_engine = BiasMitigationEngine()
    
    # Define various constraints
    constraints = [
        FairnessConstraint("demographic_parity", 0.05, "less_than", "gender"),
        FairnessConstraint("disparate_impact", 0.9, "greater_than", "gender"),
        FairnessConstraint("demographic_parity", 0.1, "less_than", "race"),
        FairnessConstraint("disparate_impact", 0.8, "greater_than", "race"),
    ]
    
    print("\nValidating fairness constraints...")
    results = mitigation_engine.validate_fairness_constraints(data, constraints)
    
    print("\nValidation Results:")
    passed = sum(results.values())
    total = len(results)
    print(f"Overall: {passed}/{total} constraints passed ({passed/total:.1%})")
    
    for constraint_desc, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {constraint_desc}: {status}")


def main():
    """Main demonstration function."""
    try:
        # Demonstrate bias detection
        bias_report, data = demonstrate_bias_detection()
        
        # Demonstrate bias mitigation
        recommendations = demonstrate_bias_mitigation(bias_report, data)
        
        # Demonstrate fairness validation
        demonstrate_fairness_validation(data)
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Automatic bias detection across multiple protected attributes")
        print("✓ Comprehensive fairness metrics calculation")
        print("✓ Intelligent mitigation strategy generation")
        print("✓ Implementation roadmap with timeline and resource estimates")
        print("✓ Fairness constraint validation")
        print("✓ Prioritized recommendations based on impact and complexity")
        
        print(f"\nBias violations detected: {len(bias_report.fairness_violations)}")
        print(f"Mitigation strategies recommended: {len(recommendations['recommended_strategies'])}")
        print(f"Implementation phases: {len(recommendations['implementation_roadmap'])}")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()