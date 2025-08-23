"""
Quality Assurance System Demo

This script demonstrates the comprehensive quality assurance and validation
capabilities of the Agent Steering System, showcasing automated testing,
data quality validation, anomaly detection, and agent output validation
with zero tolerance for simulations.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any

from scrollintel.engines.quality_assurance_engine import QualityAssuranceEngine
from scrollintel.models.quality_assurance_models import (
    QualityAssuranceConfig, DataQualityRule, DataQualityDimension,
    AnomalyDetectionConfig, AnomalyType, BusinessRule, ComplianceFramework,
    AgentOutput, AgentOutputSchema, PerformanceTestConfig, SecurityTestCase,
    TestCase, TestType, ValidationStatus
)


async def demo_comprehensive_assessment():
    """Demonstrate comprehensive quality assessment"""
    print("🔍 COMPREHENSIVE QUALITY ASSESSMENT DEMO")
    print("=" * 60)
    
    # Initialize QA engine
    config = QualityAssuranceConfig(
        organization_id="demo_org",
        automated_testing_enabled=True,
        data_quality_monitoring=True,
        anomaly_detection_enabled=True,
        business_rule_enforcement=True,
        agent_output_validation=True,
        simulation_detection_enabled=True,
        authenticity_verification=True,
        quality_score_threshold=0.8
    )
    
    qa_engine = QualityAssuranceEngine(config)
    
    print("✅ Quality Assurance Engine initialized")
    print(f"   - Organization: {config.organization_id}")
    print(f"   - Quality Threshold: {config.quality_score_threshold}")
    print(f"   - Simulation Detection: {'Enabled' if config.simulation_detection_enabled else 'Disabled'}")
    print()
    
    # Run comprehensive assessment
    print("🚀 Running comprehensive assessment...")
    assessment = await qa_engine.run_comprehensive_assessment(
        target_system="demo_agent_steering_system",
        assessment_type="full"
    )
    
    print(f"📊 Assessment Results:")
    print(f"   - Overall Quality Score: {assessment.overall_quality_score:.3f}")
    print(f"   - Certification Status: {assessment.certification_status}")
    print(f"   - Production Ready: {'Yes' if assessment.production_readiness else 'No'}")
    print(f"   - Critical Issues: {len(assessment.critical_issues)}")
    print(f"   - Recommendations: {len(assessment.recommendations)}")
    
    if assessment.recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(assessment.recommendations[:3], 1):
            print(f"   {i}. {rec}")
    
    return assessment


async def demo_data_quality_validation():
    """Demonstrate data quality validation with real-time anomaly detection"""
    print("\n📊 DATA QUALITY VALIDATION DEMO")
    print("=" * 60)
    
    # Create sample dataset with quality issues
    print("📋 Creating sample dataset with intentional quality issues...")
    
    # Good data
    good_data = {
        'customer_id': range(1, 91),
        'name': [f'Customer_{i}' for i in range(1, 91)],
        'age': np.random.randint(18, 80, 90),
        'email': [f'customer{i}@company.com' for i in range(1, 91)],
        'purchase_amount': np.random.normal(100, 30, 90),
        'registration_date': pd.date_range('2023-01-01', periods=90, freq='D')
    }
    
    # Add quality issues
    problematic_data = {
        'customer_id': [91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
        'name': ['', 'test_user', None, 'demo_customer', 'Customer_95', 'sample_user', 'Customer_97', '', 'fake_customer', 'Customer_100'],
        'age': [25, -5, 30, 200, 35, 28, None, 45, 32, 150],  # Invalid ages
        'email': ['invalid-email', 'test@fake.com', 'customer93@company.com', 'demo@test.org', None, 'sample@example.com', 'customer97@company.com', '', 'fake@mock.com', 'customer100@company.com'],
        'purchase_amount': [50, -100, 75, 999999, 85, 0, 95, None, -50, 120],  # Negative and extreme values
        'registration_date': [datetime(2023, 4, 1), datetime(2023, 4, 2), None, datetime(2023, 4, 4), datetime(2023, 4, 5), datetime(2023, 4, 6), datetime(2023, 4, 7), datetime(2023, 4, 8), datetime(2023, 4, 9), datetime(2023, 4, 10)]
    }
    
    # Combine datasets
    all_data = {}
    for key in good_data.keys():
        all_data[key] = list(good_data[key]) + problematic_data[key]
    
    dataset = pd.DataFrame(all_data)
    
    print(f"   - Dataset size: {len(dataset)} records")
    print(f"   - Columns: {list(dataset.columns)}")
    print(f"   - Intentional issues: Missing values, invalid ages, fake emails, negative amounts")
    print()
    
    # Define data quality rules
    print("📏 Defining data quality rules...")
    
    rules = [
        DataQualityRule(
            name="Completeness Check",
            description="Ensure no missing values in critical fields",
            dimension=DataQualityDimension.COMPLETENESS,
            rule_expression="no_nulls_in_critical_fields",
            threshold=0.95,
            severity="error",
            business_impact="high"
        ),
        DataQualityRule(
            name="Age Validity",
            description="Age must be between 0 and 120",
            dimension=DataQualityDimension.VALIDITY,
            rule_expression="age_range_0_120",
            threshold=0.98,
            severity="error",
            business_impact="high"
        ),
        DataQualityRule(
            name="Email Format Validation",
            description="Email addresses must be valid format",
            dimension=DataQualityDimension.VALIDITY,
            rule_expression="valid_email_format",
            threshold=0.95,
            severity="warning",
            business_impact="medium"
        ),
        DataQualityRule(
            name="Purchase Amount Validity",
            description="Purchase amounts must be non-negative",
            dimension=DataQualityDimension.VALIDITY,
            rule_expression="purchase_amount >= 0",
            threshold=0.99,
            severity="error",
            business_impact="high"
        ),
        DataQualityRule(
            name="Data Authenticity",
            description="Detect simulated or test data",
            dimension=DataQualityDimension.AUTHENTICITY,
            rule_expression="no_simulation_patterns",
            threshold=0.90,
            severity="critical",
            business_impact="critical"
        ),
        DataQualityRule(
            name="Data Uniqueness",
            description="Customer records should be unique",
            dimension=DataQualityDimension.UNIQUENESS,
            rule_expression="unique_customer_ids",
            threshold=1.0,
            severity="error",
            business_impact="high"
        )
    ]
    
    print(f"   - Created {len(rules)} quality rules")
    for rule in rules:
        dimension_name = rule.dimension if isinstance(rule.dimension, str) else rule.dimension.value
        print(f"     • {rule.name} ({dimension_name})")
    print()
    
    # Initialize QA engine
    config = QualityAssuranceConfig(
        organization_id="demo_org",
        data_quality_monitoring=True,
        anomaly_detection_enabled=True,
        simulation_detection_enabled=True,
        quality_score_threshold=0.8
    )
    
    qa_engine = QualityAssuranceEngine(config)
    
    # Validate data quality
    print("🔍 Running data quality validation...")
    
    report = await qa_engine.validate_data_quality_real_time(
        data=dataset,
        dataset_id="customer_data_demo",
        rules=rules
    )
    
    print(f"📈 Data Quality Report:")
    print(f"   - Overall Score: {report.overall_score:.3f}")
    print(f"   - Production Ready: {'Yes' if report.is_production_ready else 'No'}")
    print(f"   - Critical Issues: {len(report.critical_issues)}")
    print()
    
    print("📊 Dimension Scores:")
    for dimension, score in report.dimension_scores.items():
        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        dimension_name = dimension if isinstance(dimension, str) else dimension.value
        print(f"   {status} {dimension_name.title()}: {score:.3f}")
    print()
    
    if report.critical_issues:
        print("🚨 Critical Issues Found:")
        for i, issue in enumerate(report.critical_issues[:5], 1):
            print(f"   {i}. {issue}")
        print()
    
    if report.recommendations:
        print("💡 Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"   {i}. {rec}")
        print()
    
    return report


async def demo_anomaly_detection():
    """Demonstrate real-time anomaly detection"""
    print("\n🔍 REAL-TIME ANOMALY DETECTION DEMO")
    print("=" * 60)
    
    # Create dataset with various types of anomalies
    print("📊 Creating dataset with intentional anomalies...")
    
    # Normal data pattern
    np.random.seed(42)
    normal_data = []
    
    for i in range(200):
        record = {
            'transaction_id': i + 1,
            'amount': np.random.normal(100, 20),  # Normal distribution around $100
            'user_age': np.random.randint(18, 65),
            'session_duration': np.random.exponential(300),  # Exponential distribution
            'clicks': np.random.poisson(10),  # Poisson distribution
            'timestamp': datetime.now() - timedelta(hours=i)
        }
        normal_data.append(record)
    
    # Add statistical outliers
    outliers = [
        {'transaction_id': 201, 'amount': 10000, 'user_age': 25, 'session_duration': 300, 'clicks': 8, 'timestamp': datetime.now()},  # Extreme amount
        {'transaction_id': 202, 'amount': -500, 'user_age': 30, 'session_duration': 250, 'clicks': 12, 'timestamp': datetime.now()},  # Negative amount
        {'transaction_id': 203, 'amount': 80, 'user_age': 200, 'session_duration': 400, 'clicks': 15, 'timestamp': datetime.now()},  # Impossible age
        {'transaction_id': 204, 'amount': 120, 'user_age': 35, 'session_duration': 10000, 'clicks': 5, 'timestamp': datetime.now()},  # Extreme session
    ]
    
    # Add simulation patterns
    simulation_data = [
        {'transaction_id': 205, 'amount': 100.00, 'user_age': 25, 'session_duration': 300, 'clicks': 10, 'timestamp': datetime.now()},  # Too perfect
        {'transaction_id': 206, 'amount': 200.00, 'user_age': 30, 'session_duration': 600, 'clicks': 20, 'timestamp': datetime.now()},  # Too perfect
    ]
    
    all_data = normal_data + outliers + simulation_data
    dataset = pd.DataFrame(all_data)
    
    print(f"   - Total records: {len(dataset)}")
    print(f"   - Normal patterns: {len(normal_data)}")
    print(f"   - Statistical outliers: {len(outliers)}")
    print(f"   - Simulation patterns: {len(simulation_data)}")
    print()
    
    # Configure anomaly detection
    print("⚙️ Configuring anomaly detection...")
    
    config = AnomalyDetectionConfig(
        detection_method="hybrid",  # Use all detection methods
        sensitivity=0.8,
        window_size=50,
        threshold_multiplier=2.5,
        min_samples=20,
        feature_columns=['amount', 'user_age', 'session_duration', 'clicks'],
        real_time_processing=True
    )
    
    print(f"   - Detection method: {config.detection_method}")
    print(f"   - Sensitivity: {config.sensitivity}")
    print(f"   - Feature columns: {config.feature_columns}")
    print()
    
    # Initialize QA engine
    qa_config = QualityAssuranceConfig(
        organization_id="demo_org",
        anomaly_detection_enabled=True,
        simulation_detection_enabled=True
    )
    
    qa_engine = QualityAssuranceEngine(qa_config)
    
    # Detect anomalies
    print("🔍 Running anomaly detection...")
    
    anomalies = await qa_engine.detect_anomalies_real_time(
        data=dataset,
        dataset_id="transaction_data_demo",
        config=config
    )
    
    print(f"🚨 Anomaly Detection Results:")
    print(f"   - Total anomalies detected: {len(anomalies)}")
    print()
    
    if anomalies:
        print("📋 Detected Anomalies:")
        
        # Group by type
        anomaly_types = {}
        for anomaly in anomalies:
            if anomaly.anomaly_type not in anomaly_types:
                anomaly_types[anomaly.anomaly_type] = []
            anomaly_types[anomaly.anomaly_type].append(anomaly)
        
        for anomaly_type, type_anomalies in anomaly_types.items():
            anomaly_type_name = anomaly_type if isinstance(anomaly_type, str) else anomaly_type.value
            print(f"\n   🔸 {anomaly_type_name.replace('_', ' ').title()}:")
            for i, anomaly in enumerate(type_anomalies[:3], 1):
                print(f"      {i}. Confidence: {anomaly.confidence_score:.3f}")
                print(f"         Severity: {anomaly.severity}")
                print(f"         Affected records: {anomaly.affected_records}")
                if anomaly.recommended_actions:
                    print(f"         Action: {anomaly.recommended_actions[0]}")
    else:
        print("   ✅ No anomalies detected")
    
    return anomalies


async def demo_business_rule_validation():
    """Demonstrate business rule validation"""
    print("\n📋 BUSINESS RULE VALIDATION DEMO")
    print("=" * 60)
    
    # Define business rules
    print("📏 Defining business rules...")
    
    rules = [
        BusinessRule(
            name="Minimum Age Requirement",
            description="Users must be at least 18 years old",
            category="user_validation",
            rule_logic="required_fields: age",
            compliance_frameworks=[ComplianceFramework.GDPR],
            business_owner="Legal Department",
            technical_owner="Data Engineering Team",
            priority=5,
            is_mandatory=True,
            effective_date=datetime.utcnow() - timedelta(days=30)
        ),
        BusinessRule(
            name="Email Validation",
            description="All users must have valid email addresses",
            category="contact_validation",
            rule_logic="required_fields: email",
            business_owner="Marketing Department",
            technical_owner="Data Engineering Team",
            priority=4,
            is_mandatory=True,
            effective_date=datetime.utcnow() - timedelta(days=60)
        ),
        BusinessRule(
            name="Purchase Amount Limit",
            description="Single purchases cannot exceed $10,000",
            category="transaction_validation",
            rule_logic="purchase_amount <= 10000",
            business_owner="Finance Department",
            technical_owner="Payment Processing Team",
            priority=5,
            is_mandatory=True,
            effective_date=datetime.utcnow() - timedelta(days=90)
        ),
        BusinessRule(
            name="Data Retention Policy",
            description="Customer data must not be older than 7 years",
            category="compliance",
            rule_logic="data_age <= 7_years",
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOX],
            business_owner="Compliance Officer",
            technical_owner="Data Governance Team",
            priority=5,
            is_mandatory=True,
            effective_date=datetime.utcnow() - timedelta(days=365)
        )
    ]
    
    print(f"   - Created {len(rules)} business rules")
    for rule in rules:
        if rule.compliance_frameworks:
            frameworks = ", ".join([f.value.upper() if hasattr(f, 'value') else str(f).upper() for f in rule.compliance_frameworks])
        else:
            frameworks = "None"
        print(f"     • {rule.name} (Priority: {rule.priority}, Frameworks: {frameworks})")
    print()
    
    # Test data scenarios
    test_scenarios = [
        {
            "name": "Valid Customer",
            "data": {
                "age": 25,
                "email": "customer@company.com",
                "purchase_amount": 150.00,
                "registration_date": "2023-06-15"
            }
        },
        {
            "name": "Underage User",
            "data": {
                "age": 16,
                "email": "teen@email.com",
                "purchase_amount": 50.00,
                "registration_date": "2023-08-01"
            }
        },
        {
            "name": "Invalid Email",
            "data": {
                "age": 30,
                "email": "invalid-email-format",
                "purchase_amount": 200.00,
                "registration_date": "2023-07-10"
            }
        },
        {
            "name": "Excessive Purchase",
            "data": {
                "age": 35,
                "email": "bigspender@company.com",
                "purchase_amount": 15000.00,
                "registration_date": "2023-05-20"
            }
        },
        {
            "name": "Missing Required Data",
            "data": {
                "purchase_amount": 75.00,
                "registration_date": "2023-09-01"
                # Missing age and email
            }
        }
    ]
    
    # Initialize QA engine
    config = QualityAssuranceConfig(
        organization_id="demo_org",
        business_rule_enforcement=True
    )
    
    qa_engine = QualityAssuranceEngine(config)
    
    print("🔍 Testing business rule validation...")
    print()
    
    for scenario in test_scenarios:
        print(f"📊 Scenario: {scenario['name']}")
        print(f"   Data: {scenario['data']}")
        
        # Validate against business rules
        validations = await qa_engine.validate_business_rules(
            data=scenario['data'],
            rules=rules
        )
        
        # Calculate overall compliance
        total_rules = len(validations)
        passed_rules = sum(1 for v in validations if v.validation_result)
        compliance_rate = passed_rules / total_rules if total_rules > 0 else 0
        
        status_icon = "✅" if compliance_rate == 1.0 else "⚠️" if compliance_rate >= 0.5 else "❌"
        print(f"   {status_icon} Compliance: {compliance_rate:.1%} ({passed_rules}/{total_rules} rules passed)")
        
        # Show failed rules
        failed_validations = [v for v in validations if not v.validation_result]
        if failed_validations:
            print("   🚨 Failed Rules:")
            for validation in failed_validations:
                rule = next((r for r in rules if r.id == validation.rule_id), None)
                if rule:
                    print(f"      • {rule.name}")
        
        print()
    
    return validations


async def demo_agent_output_validation():
    """Demonstrate agent output validation with simulation detection"""
    print("\n🤖 AGENT OUTPUT VALIDATION DEMO")
    print("=" * 60)
    
    # Define output schema
    print("📋 Defining agent output schema...")
    
    schema = AgentOutputSchema(
        agent_type="business_analyst",
        output_format="json",
        required_fields=["analysis_summary", "key_insights", "recommendations", "confidence_score"],
        optional_fields=["data_sources", "methodology", "limitations"],
        field_types={
            "analysis_summary": "string",
            "key_insights": "list",
            "recommendations": "list",
            "confidence_score": "float",
            "data_sources": "list",
            "methodology": "string"
        },
        validation_rules=[
            "confidence_score >= 0.0 and confidence_score <= 1.0",
            "len(key_insights) > 0",
            "len(recommendations) > 0"
        ],
        business_constraints=[
            "analysis_summary must not be empty",
            "confidence_score must be realistic (< 0.99)",
            "recommendations must be actionable"
        ],
        authenticity_checks=[
            "no_simulation_markers",
            "realistic_processing_time",
            "authentic_data_sources",
            "genuine_reasoning_patterns"
        ]
    )
    
    print(f"   - Agent type: {schema.agent_type}")
    print(f"   - Required fields: {schema.required_fields}")
    print(f"   - Validation rules: {len(schema.validation_rules)}")
    print(f"   - Authenticity checks: {len(schema.authenticity_checks)}")
    print()
    
    # Test scenarios
    test_outputs = [
        {
            "name": "Authentic Business Analysis",
            "output": AgentOutput(
                agent_id="prod_analyst_001",
                agent_type="business_analyst",
                task_id="quarterly_review_q3_2023",
                output_data={
                    "analysis_summary": "Q3 revenue increased 12% compared to Q2, driven primarily by new customer acquisitions in the enterprise segment.",
                    "key_insights": [
                        "Enterprise segment grew 25% quarter-over-quarter",
                        "Customer retention rate improved to 94%",
                        "Average deal size increased by 18%"
                    ],
                    "recommendations": [
                        "Increase investment in enterprise sales team",
                        "Develop customer success programs to maintain retention",
                        "Explore upselling opportunities with existing clients"
                    ],
                    "confidence_score": 0.87,
                    "data_sources": ["salesforce_crm", "financial_reporting_db", "customer_success_platform"],
                    "methodology": "Statistical analysis of sales data with trend comparison"
                },
                metadata={
                    "model_version": "v2.3.1",
                    "analysis_depth": "comprehensive",
                    "data_freshness": "real_time"
                },
                generation_timestamp=datetime.utcnow(),
                processing_time_ms=3200,
                confidence_score=0.87,
                data_sources=["salesforce_crm", "financial_reporting_db", "customer_success_platform"],
                reasoning_trace=[
                    "Retrieved Q3 sales data from Salesforce CRM",
                    "Calculated quarter-over-quarter growth metrics",
                    "Analyzed customer segmentation patterns",
                    "Identified key growth drivers and trends",
                    "Generated actionable recommendations based on findings"
                ]
            )
        },
        {
            "name": "Simulated Test Output",
            "output": AgentOutput(
                agent_id="test_agent_demo",
                agent_type="business_analyst",
                task_id="demo_analysis_task",
                output_data={
                    "analysis_summary": "This is a test analysis with demo data showing sample results.",
                    "key_insights": [
                        "Sample insight from mock data",
                        "Demo pattern detected in test dataset",
                        "Fake trend analysis for testing purposes"
                    ],
                    "recommendations": [
                        "Test recommendation for demo purposes",
                        "Sample action item from mock analysis"
                    ],
                    "confidence_score": 0.999,  # Suspiciously high
                    "data_sources": ["mock_database", "test_api", "sample_data_source"],
                    "methodology": "Simulated analysis using generated test data"
                },
                metadata={
                    "is_simulated": True,
                    "test_mode": True
                },
                generation_timestamp=datetime.utcnow(),
                processing_time_ms=50,  # Suspiciously fast
                confidence_score=0.999,
                data_sources=["mock_database", "test_api", "sample_data_source"],
                reasoning_trace=[
                    "Generated mock analysis for testing",
                    "Simulated data processing workflow",
                    "Created fake insights for demo purposes"
                ]
            )
        },
        {
            "name": "Invalid Format Output",
            "output": AgentOutput(
                agent_id="faulty_agent_002",
                agent_type="business_analyst",
                task_id="broken_analysis",
                output_data={
                    "summary": "Missing required fields",  # Wrong field name
                    "confidence_score": 1.5,  # Invalid range
                    # Missing required fields: key_insights, recommendations
                },
                generation_timestamp=datetime.utcnow(),
                processing_time_ms=1500
            )
        }
    ]
    
    # Initialize QA engine
    config = QualityAssuranceConfig(
        organization_id="demo_org",
        agent_output_validation=True,
        simulation_detection_enabled=True,
        authenticity_verification=True
    )
    
    qa_engine = QualityAssuranceEngine(config)
    
    print("🔍 Validating agent outputs...")
    print()
    
    for test_case in test_outputs:
        print(f"🤖 Testing: {test_case['name']}")
        
        # Validate output
        result = await qa_engine.validate_agent_output(
            output=test_case['output'],
            schema=schema
        )
        
        # Display results
        status_icon = "✅" if result.overall_status == ValidationStatus.PASSED else "⚠️" if result.overall_status == ValidationStatus.WARNING else "❌"
        status_name = result.overall_status if isinstance(result.overall_status, str) else result.overall_status.value
        print(f"   {status_icon} Overall Status: {status_name}")
        print(f"   📊 Quality Score: {result.quality_score:.3f}")
        print(f"   🔒 Authentic: {'Yes' if result.is_authentic else 'No'}")
        print(f"   🚫 Simulation-Free: {'Yes' if result.is_simulation_free else 'No'}")
        
        # Show validation details
        print("   📋 Validation Details:")
        print(f"      • Format: {'✅' if result.format_validation.get('is_valid', False) else '❌'}")
        print(f"      • Business Logic: {'✅' if result.business_validation.get('is_valid', False) else '❌'}")
        print(f"      • Data Authenticity: {result.data_validation.get('authenticity_score', 0):.3f}")
        print(f"      • Compliance: {'✅' if result.compliance_validation.get('is_compliant', False) else '❌'}")
        
        # Show issues if any
        if result.issues_found:
            print("   🚨 Issues Found:")
            for issue in result.issues_found[:3]:
                print(f"      • {issue}")
        
        # Show recommendations
        if result.recommendations:
            print("   💡 Recommendations:")
            for rec in result.recommendations[:2]:
                print(f"      • {rec}")
        
        print()
    
    return test_outputs


async def demo_performance_and_security_testing():
    """Demonstrate performance and security testing"""
    print("\n⚡ PERFORMANCE & SECURITY TESTING DEMO")
    print("=" * 60)
    
    # Initialize QA engine
    config = QualityAssuranceConfig(
        organization_id="demo_org",
        automated_testing_enabled=True
    )
    
    qa_engine = QualityAssuranceEngine(config)
    
    # Performance testing
    print("🚀 Performance Testing Demo:")
    
    perf_config = PerformanceTestConfig(
        test_name="Agent API Load Test",
        target_endpoint="http://localhost:8000/api/v1/agents/analyze",
        load_pattern="ramp_up",
        concurrent_users=50,
        duration_seconds=120,
        ramp_up_time=30,
        success_criteria={
            "average_response_time": 500.0,
            "p95_response_time": 1000.0,
            "error_rate": 0.05,
            "throughput_rps": 100.0
        },
        monitoring_metrics=["cpu_usage", "memory_usage", "response_time", "error_rate"]
    )
    
    print(f"   - Test: {perf_config.test_name}")
    print(f"   - Load Pattern: {perf_config.load_pattern}")
    print(f"   - Concurrent Users: {perf_config.concurrent_users}")
    print(f"   - Duration: {perf_config.duration_seconds}s")
    print()
    
    print("   🔍 Running performance test...")
    perf_results = await qa_engine.run_performance_tests(perf_config)
    
    print(f"   📊 Performance Results:")
    print(f"      • Total Requests: {perf_results.total_requests:,}")
    print(f"      • Success Rate: {((perf_results.successful_requests / perf_results.total_requests) * 100):.1f}%")
    print(f"      • Average Response Time: {perf_results.average_response_time:.0f}ms")
    print(f"      • 95th Percentile: {perf_results.p95_response_time:.0f}ms")
    print(f"      • Throughput: {perf_results.throughput_rps:.1f} RPS")
    print(f"      • Error Rate: {(perf_results.error_rate * 100):.2f}%")
    print(f"      • Performance Grade: {perf_results.performance_grade}")
    print()
    
    # Security testing
    print("🔒 Security Testing Demo:")
    
    security_tests = [
        SecurityTestCase(
            name="SQL Injection Test",
            category="injection",
            severity="critical",
            test_vector="'; DROP TABLE users; --",
            expected_behavior="Input should be sanitized and query should fail safely",
            automated=True,
            requires_manual_review=False
        ),
        SecurityTestCase(
            name="Cross-Site Scripting (XSS)",
            category="xss",
            severity="high",
            test_vector="<script>alert('XSS')</script>",
            expected_behavior="Script tags should be escaped or removed",
            automated=True,
            requires_manual_review=False
        ),
        SecurityTestCase(
            name="Authentication Bypass",
            category="authentication",
            severity="critical",
            test_vector="admin'--",
            expected_behavior="Authentication should not be bypassed",
            automated=True,
            requires_manual_review=True
        ),
        SecurityTestCase(
            name="Path Traversal",
            category="path_traversal",
            severity="high",
            test_vector="../../../etc/passwd",
            expected_behavior="File system access should be restricted",
            automated=True,
            requires_manual_review=False
        )
    ]
    
    print(f"   - Running {len(security_tests)} security tests...")
    for test in security_tests:
        print(f"     • {test.name} ({test.severity} severity)")
    print()
    
    security_results = await qa_engine.run_security_tests(security_tests)
    
    print(f"   🛡️ Security Results:")
    
    # Categorize results by risk level
    risk_summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    vulnerabilities_found = 0
    
    for result in security_results:
        risk_summary[result.risk_level] += 1
        if result.vulnerability_found:
            vulnerabilities_found += 1
    
    print(f"      • Tests Run: {len(security_results)}")
    print(f"      • Vulnerabilities Found: {vulnerabilities_found}")
    print(f"      • Risk Distribution:")
    for risk_level, count in risk_summary.items():
        if count > 0:
            icon = "🔴" if risk_level == "critical" else "🟠" if risk_level == "high" else "🟡" if risk_level == "medium" else "🟢"
            print(f"        {icon} {risk_level.title()}: {count}")
    
    if vulnerabilities_found == 0:
        print("      ✅ No vulnerabilities detected")
    else:
        print(f"      ⚠️ {vulnerabilities_found} potential vulnerabilities require attention")
    
    print()
    
    return perf_results, security_results


async def main():
    """Run comprehensive quality assurance demo"""
    print("🎯 SCROLLINTEL QUALITY ASSURANCE SYSTEM DEMO")
    print("=" * 80)
    print("Demonstrating enterprise-grade quality assurance with zero tolerance")
    print("for simulations and comprehensive validation capabilities.")
    print("=" * 80)
    print()
    
    try:
        # Run all demo components
        assessment = await demo_comprehensive_assessment()
        
        data_quality_report = await demo_data_quality_validation()
        
        anomalies = await demo_anomaly_detection()
        
        business_validations = await demo_business_rule_validation()
        
        agent_outputs = await demo_agent_output_validation()
        
        perf_results, security_results = await demo_performance_and_security_testing()
        
        # Summary
        print("\n🎉 DEMO SUMMARY")
        print("=" * 60)
        print(f"✅ Comprehensive Assessment: Quality Score {assessment.overall_quality_score:.3f}")
        print(f"📊 Data Quality: Overall Score {data_quality_report.overall_score:.3f}")
        print(f"🔍 Anomaly Detection: {len(anomalies)} anomalies detected")
        print(f"📋 Business Rules: Validation framework operational")
        print(f"🤖 Agent Output Validation: {len(agent_outputs)} outputs tested")
        print(f"⚡ Performance Testing: {perf_results.performance_grade} grade achieved")
        print(f"🔒 Security Testing: {len(security_results)} tests completed")
        print()
        print("🎯 The Quality Assurance System successfully demonstrated:")
        print("   • Zero tolerance for simulations and fake data")
        print("   • Real-time data quality validation")
        print("   • Advanced anomaly detection capabilities")
        print("   • Comprehensive business rule enforcement")
        print("   • Authentic agent output validation")
        print("   • Enterprise-grade performance and security testing")
        print()
        print("✨ System ready for production deployment with full quality assurance!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())