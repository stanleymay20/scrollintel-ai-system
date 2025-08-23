"""
Demo script for Data Quality Monitoring System
Demonstrates the complete data quality monitoring workflow
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from scrollintel.models.data_quality_models import Base, QualityRuleType, Severity
from scrollintel.engines.data_quality_monitor import DataQualityMonitor
from scrollintel.engines.anomaly_detector import AnomalyDetector
from scrollintel.engines.quality_alerting import QualityAlertManager, RealTimeQualityMonitor, AlertConfig, AlertChannel
from scrollintel.engines.data_profiler import DataProfiler, ProfilingConfig

def create_sample_data():
    """Create sample dataset with various quality issues"""
    np.random.seed(42)
    
    # Create base dataset
    n_records = 5000
    data = pd.DataFrame({
        'customer_id': range(1, n_records + 1),
        'first_name': [f'Customer_{i}' for i in range(1, n_records + 1)],
        'last_name': [f'Lastname_{i}' for i in range(1, n_records + 1)],
        'email': [f'customer{i}@example.com' for i in range(1, n_records + 1)],
        'phone': [f'+1-555-{str(i).zfill(7)}' for i in range(1, n_records + 1)],
        'age': np.random.normal(35, 12, n_records).astype(int),
        'annual_income': np.random.normal(65000, 20000, n_records),
        'credit_score': np.random.normal(720, 80, n_records).astype(int),
        'account_balance': np.random.normal(5000, 15000, n_records),
        'registration_date': pd.date_range('2020-01-01', periods=n_records, freq='H'),
        'is_active': np.random.choice([True, False], n_records, p=[0.85, 0.15]),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_records, p=[0.2, 0.6, 0.2])
    })
    
    # Introduce quality issues for demonstration
    
    # 1. Missing values (completeness issues)
    missing_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
    data.loc[missing_indices, 'email'] = None
    
    missing_phone_indices = np.random.choice(n_records, size=int(n_records * 0.03), replace=False)
    data.loc[missing_phone_indices, 'phone'] = None
    
    # 2. Invalid data (validity issues)
    invalid_email_indices = np.random.choice(n_records, size=50, replace=False)
    data.loc[invalid_email_indices, 'email'] = 'invalid_email_format'
    
    invalid_age_indices = np.random.choice(n_records, size=30, replace=False)
    data.loc[invalid_age_indices, 'age'] = -1  # Negative ages
    
    invalid_credit_indices = np.random.choice(n_records, size=25, replace=False)
    data.loc[invalid_credit_indices, 'credit_score'] = 950  # Impossible credit scores
    
    # 3. Duplicate records (uniqueness issues)
    duplicate_indices = np.random.choice(n_records, size=20, replace=False)
    for idx in duplicate_indices:
        if idx < n_records - 1:
            data.loc[idx + 1, 'customer_id'] = data.loc[idx, 'customer_id']
    
    # 4. Outliers (statistical anomalies)
    outlier_indices = np.random.choice(n_records, size=15, replace=False)
    data.loc[outlier_indices, 'annual_income'] = 500000  # Very high incomes
    
    outlier_balance_indices = np.random.choice(n_records, size=10, replace=False)
    data.loc[outlier_balance_indices, 'account_balance'] = -50000  # Large negative balances
    
    # 5. Inconsistencies
    inconsistent_indices = np.random.choice(n_records, size=40, replace=False)
    data.loc[inconsistent_indices, 'customer_segment'] = 'Premium'
    data.loc[inconsistent_indices, 'annual_income'] = 25000  # Low income but premium segment
    
    return data

def setup_database():
    """Setup in-memory database for demo"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

async def demonstrate_data_quality_monitoring():
    """Demonstrate complete data quality monitoring workflow"""
    print("🚀 Starting Data Quality Monitoring Demo")
    print("=" * 60)
    
    # Setup
    db_session = setup_database()
    sample_data = create_sample_data()
    
    print(f"📊 Created sample dataset with {len(sample_data)} records")
    print(f"   Columns: {list(sample_data.columns)}")
    print()
    
    # Step 1: Data Profiling
    print("📈 Step 1: Creating Data Profiles")
    print("-" * 40)
    
    profiler = DataProfiler(db_session, ProfilingConfig(sample_size=1000))
    profiles = profiler.create_comprehensive_profile(sample_data, "customers", "demo_pipeline")
    
    print(f"✅ Created {len(profiles)} column profiles")
    
    # Show some profile insights
    for profile in profiles[:3]:  # Show first 3 profiles
        print(f"   📋 {profile.column_name}:")
        print(f"      - Data type: {profile.data_type}")
        print(f"      - Completeness: {profile.completeness_score:.1f}%")
        print(f"      - Unique values: {profile.unique_count}/{profile.record_count}")
        if profile.mean_value:
            print(f"      - Mean: {profile.mean_value:.2f}")
        if profile.format_patterns:
            patterns = list(profile.format_patterns.keys())[:2]
            print(f"      - Detected patterns: {patterns}")
        print()
    
    # Step 2: Establish Quality Baseline
    print("🎯 Step 2: Establishing Quality Baseline")
    print("-" * 40)
    
    baseline = profiler.establish_quality_baseline(profiles, "customers", "demo_pipeline")
    
    print(f"✅ Quality Baseline Established:")
    print(f"   - Completeness baseline: {baseline.completeness_baseline:.1f}%")
    print(f"   - Validity baseline: {baseline.validity_baseline:.1f}%")
    print(f"   - Uniqueness baseline: {baseline.uniqueness_baseline:.1f}%")
    print(f"   - Recommended rules: {len(baseline.recommended_rules)}")
    print()
    
    # Step 3: Create Quality Rules
    print("📏 Step 3: Creating Quality Rules")
    print("-" * 40)
    
    monitor = DataQualityMonitor(db_session)
    created_rules = []
    
    # Create rules from recommendations
    for rule_config in baseline.recommended_rules[:8]:  # Create first 8 rules
        try:
            rule_id = monitor.create_quality_rule(rule_config)
            created_rules.append(rule_id)
            print(f"   ✅ Created rule: {rule_config['name']}")
        except Exception as e:
            print(f"   ❌ Failed to create rule: {rule_config['name']} - {str(e)}")
    
    print(f"\n✅ Created {len(created_rules)} quality rules")
    print()
    
    # Step 4: Validate Data Quality
    print("🔍 Step 4: Validating Data Quality")
    print("-" * 40)
    
    # Get created rules
    from scrollintel.models.data_quality_models import QualityRule
    rules = db_session.query(QualityRule).filter(QualityRule.id.in_(created_rules)).all()
    
    # Validate data
    reports = monitor.validate_data_batch(sample_data, rules, "demo_execution_001")
    
    print(f"✅ Validation completed - {len(reports)} reports generated")
    
    # Show validation results
    passed_count = sum(1 for r in reports if r.status.value == 'passed')
    failed_count = len(reports) - passed_count
    
    print(f"   📊 Results: {passed_count} passed, {failed_count} failed")
    
    for report in reports:
        status_icon = "✅" if report.status.value == 'passed' else "❌"
        rule = next(r for r in rules if r.id == report.rule_id)
        print(f"   {status_icon} {rule.name}: {report.score:.1f}% "
              f"({report.records_failed}/{report.records_checked} failed)")
    
    print()
    
    # Step 5: Anomaly Detection
    print("🔍 Step 5: Detecting Anomalies")
    print("-" * 40)
    
    detector = AnomalyDetector(db_session)
    
    # Detect anomalies in income column (has outliers)
    income_profile = next(p for p in profiles if p.column_name == 'annual_income')
    anomalies = detector.detect_anomalies(
        sample_data, 
        income_profile, 
        ['statistical', 'isolation_forest']
    )
    
    print(f"✅ Anomaly detection completed")
    print(f"   🚨 Found {len(anomalies)} anomalies in annual_income")
    
    # Show some anomalies
    for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
        print(f"   {i+1}. Record {anomaly.get('record_id')}: "
              f"{anomaly.get('anomaly_type')} "
              f"(confidence: {anomaly.get('confidence_score', 0):.2f})")
    
    print()
    
    # Step 6: Quality Metrics
    print("📊 Step 6: Quality Metrics")
    print("-" * 40)
    
    time_range = (datetime.utcnow() - timedelta(hours=1), datetime.utcnow())
    metrics = monitor.get_quality_metrics("demo_pipeline", time_range)
    
    print(f"✅ Quality Metrics for demo_pipeline:")
    print(f"   - Total checks: {metrics['total_checks']}")
    print(f"   - Passed checks: {metrics['passed_checks']}")
    print(f"   - Failed checks: {metrics['failed_checks']}")
    print(f"   - Average score: {metrics['average_score']:.1f}%")
    print(f"   - Success rate: {metrics['passed_checks']/metrics['total_checks']*100:.1f}%")
    print()
    
    # Step 7: Real-time Monitoring Setup
    print("⚡ Step 7: Real-time Monitoring")
    print("-" * 40)
    
    # Setup alert manager
    alert_config = AlertConfig(
        channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL],
        recipients=['admin@company.com'],
        severity_threshold=Severity.MEDIUM,
        cooldown_minutes=5
    )
    
    alert_manager = QualityAlertManager(db_session, alert_config)
    real_time_monitor = RealTimeQualityMonitor(db_session, alert_manager)
    
    print("✅ Real-time monitoring configured")
    print(f"   - Alert channels: {[c.value for c in alert_config.channels]}")
    print(f"   - Severity threshold: {alert_config.severity_threshold.value}")
    print(f"   - Cooldown period: {alert_config.cooldown_minutes} minutes")
    
    # Simulate monitoring for a short period
    print("\n🔄 Starting monitoring simulation...")
    
    try:
        # Start monitoring
        await real_time_monitor.start_monitoring(['demo_pipeline'])
        
        # Let it run for a few seconds
        await asyncio.sleep(3)
        
        # Get monitoring status
        status = real_time_monitor.get_monitoring_status()
        print(f"   📡 Monitoring status: {status}")
        
        # Get real-time metrics
        rt_metrics = real_time_monitor.get_real_time_metrics()
        print(f"   📊 Real-time metrics:")
        print(f"      - Total checks: {rt_metrics.total_checks}")
        print(f"      - Active anomalies: {rt_metrics.active_anomalies}")
        print(f"      - Critical alerts: {rt_metrics.critical_alerts}")
        print(f"      - Last check: {rt_metrics.last_check_time}")
        
        # Stop monitoring
        await real_time_monitor.stop_monitoring()
        print("   ⏹️  Monitoring stopped")
        
    except Exception as e:
        print(f"   ⚠️  Monitoring simulation error: {str(e)}")
    
    print()
    
    # Step 8: Data Drift Detection
    print("📈 Step 8: Data Drift Detection")
    print("-" * 40)
    
    # Create modified data to simulate drift
    drifted_data = sample_data.copy()
    drifted_data['annual_income'] = drifted_data['annual_income'] * 1.5  # 50% increase
    drifted_data['age'] = drifted_data['age'] + 5  # Age everyone by 5 years
    
    # Detect drift
    drift_results = []
    for column in ['annual_income', 'age']:
        profile = next(p for p in profiles if p.column_name == column)
        drift_result = detector.detect_data_drift(drifted_data, profile, column)
        drift_results.append((column, drift_result))
    
    print("✅ Data drift analysis completed:")
    for column, result in drift_results:
        drift_status = "🚨 DRIFT DETECTED" if result['drift_detected'] else "✅ No drift"
        print(f"   {column}: {drift_status}")
        if result['drift_detected']:
            print(f"      - Drift type: {result['drift_type']}")
            print(f"      - Drift score: {result['drift_score']:.3f}")
    
    print()
    
    # Summary
    print("🎉 Demo Summary")
    print("=" * 60)
    print(f"✅ Profiled {len(profiles)} columns")
    print(f"✅ Created {len(created_rules)} quality rules")
    print(f"✅ Generated {len(reports)} validation reports")
    print(f"✅ Detected {len(anomalies)} anomalies")
    print(f"✅ Identified data drift in {sum(1 for _, r in drift_results if r['drift_detected'])} columns")
    print(f"✅ Configured real-time monitoring")
    
    overall_quality = metrics['average_score'] if metrics['total_checks'] > 0 else 0
    print(f"\n📊 Overall Data Quality Score: {overall_quality:.1f}%")
    
    if overall_quality >= 90:
        print("🟢 Excellent data quality!")
    elif overall_quality >= 75:
        print("🟡 Good data quality with room for improvement")
    elif overall_quality >= 60:
        print("🟠 Moderate data quality - attention needed")
    else:
        print("🔴 Poor data quality - immediate action required")
    
    print("\n🏁 Data Quality Monitoring Demo Complete!")
    
    # Cleanup
    db_session.close()

def demonstrate_api_usage():
    """Demonstrate API usage examples"""
    print("\n🌐 API Usage Examples")
    print("=" * 60)
    
    print("📝 Example API calls for data quality monitoring:")
    print()
    
    # Quality Rules API
    print("1. Create Quality Rule:")
    print("   POST /api/v1/data-quality/rules")
    print("   {")
    print('     "name": "Email completeness check",')
    print('     "rule_type": "completeness",')
    print('     "target_table": "customers",')
    print('     "target_column": "email",')
    print('     "threshold_value": 0.95,')
    print('     "severity": "medium"')
    print("   }")
    print()
    
    # Data Validation API
    print("2. Validate Data:")
    print("   POST /api/v1/data-quality/validate")
    print("   {")
    print('     "data": [...],')
    print('     "rule_ids": ["rule-id-1", "rule-id-2"]')
    print("   }")
    print()
    
    # Data Profiling API
    print("3. Create Data Profile:")
    print("   POST /api/v1/data-quality/profile")
    print("   {")
    print('     "data": [...],')
    print('     "table_name": "customers"')
    print("   }")
    print()
    
    # Anomaly Detection API
    print("4. Detect Anomalies:")
    print("   POST /api/v1/data-quality/anomalies/detect")
    print("   {")
    print('     "data": [...],')
    print('     "profile_id": "profile-id",')
    print('     "detection_methods": ["statistical", "isolation_forest"]')
    print("   }")
    print()
    
    # Quality Metrics API
    print("5. Get Quality Metrics:")
    print("   GET /api/v1/data-quality/metrics/{pipeline_id}?hours=24")
    print()
    
    # Alerts API
    print("6. List Quality Alerts:")
    print("   GET /api/v1/data-quality/alerts?severity=high&is_resolved=false")
    print()

if __name__ == "__main__":
    print("🔍 Data Quality Monitoring System Demo")
    print("=====================================")
    print()
    
    # Run the main demo
    asyncio.run(demonstrate_data_quality_monitoring())
    
    # Show API examples
    demonstrate_api_usage()
    
    print("\n💡 Next Steps:")
    print("- Integrate with your data pipelines")
    print("- Configure real-time monitoring")
    print("- Set up alerting channels")
    print("- Customize quality rules for your data")
    print("- Monitor data quality dashboards")
    print("\n🚀 Happy monitoring!")