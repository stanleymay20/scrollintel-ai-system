"""
Demo script for AI Data Readiness Platform - Drift Monitoring and Alerting System.

This script demonstrates the comprehensive drift detection and alerting capabilities
including statistical drift detection, alert management, and notification systems.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from ai_data_readiness.engines.drift_monitor import DriftMonitor
from ai_data_readiness.engines.alert_manager import (
    AlertManager, NotificationChannel, NotificationConfig, AlertRule,
    EscalationRule, EscalationLevel, NotificationTemplate
)
from ai_data_readiness.models.drift_models import (
    DriftThresholds, AlertSeverity, DriftType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_datasets():
    """Create sample datasets for drift monitoring demonstration."""
    np.random.seed(42)
    
    # Reference dataset (baseline)
    reference_data = pd.DataFrame({
        'customer_age': np.random.normal(35, 10, 5000),
        'income': np.random.normal(50000, 15000, 5000),
        'credit_score': np.random.normal(650, 100, 5000),
        'account_type': np.random.choice(['checking', 'savings', 'premium'], 5000, p=[0.6, 0.3, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 5000, p=[0.25, 0.25, 0.25, 0.25]),
        'transaction_frequency': np.random.poisson(10, 5000)
    })
    
    # Current dataset with no drift (similar distribution)
    np.random.seed(43)
    current_no_drift = pd.DataFrame({
        'customer_age': np.random.normal(35, 10, 5000),
        'income': np.random.normal(50000, 15000, 5000),
        'credit_score': np.random.normal(650, 100, 5000),
        'account_type': np.random.choice(['checking', 'savings', 'premium'], 5000, p=[0.6, 0.3, 0.1]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 5000, p=[0.25, 0.25, 0.25, 0.25]),
        'transaction_frequency': np.random.poisson(10, 5000)
    })
    
    # Current dataset with moderate drift
    np.random.seed(44)
    current_moderate_drift = pd.DataFrame({
        'customer_age': np.random.normal(38, 12, 5000),  # Age shift
        'income': np.random.normal(52000, 16000, 5000),  # Income increase
        'credit_score': np.random.normal(645, 105, 5000),  # Slight credit score change
        'account_type': np.random.choice(['checking', 'savings', 'premium'], 5000, p=[0.5, 0.35, 0.15]),  # Distribution shift
        'region': np.random.choice(['north', 'south', 'east', 'west'], 5000, p=[0.3, 0.3, 0.2, 0.2]),  # Regional shift
        'transaction_frequency': np.random.poisson(12, 5000)  # Frequency increase
    })
    
    # Current dataset with high drift
    np.random.seed(45)
    current_high_drift = pd.DataFrame({
        'customer_age': np.random.normal(42, 15, 5000),  # Significant age shift
        'income': np.random.normal(60000, 20000, 5000),  # Major income increase
        'credit_score': np.random.normal(620, 120, 5000),  # Credit score decline
        'account_type': np.random.choice(['checking', 'savings', 'premium'], 5000, p=[0.4, 0.4, 0.2]),  # Major distribution shift
        'region': np.random.choice(['north', 'south', 'east', 'west'], 5000, p=[0.4, 0.35, 0.15, 0.1]),  # Significant regional shift
        'transaction_frequency': np.random.poisson(15, 5000)  # High frequency increase
    })
    
    return reference_data, current_no_drift, current_moderate_drift, current_high_drift


async def demonstrate_drift_detection():
    """Demonstrate drift detection capabilities."""
    print("🔍 AI Data Readiness Platform - Drift Detection Demo")
    print("=" * 60)
    
    # Create sample datasets
    reference_data, no_drift_data, moderate_drift_data, high_drift_data = create_sample_datasets()
    
    # Initialize drift monitor with custom thresholds
    custom_thresholds = DriftThresholds(
        low_threshold=0.1,
        medium_threshold=0.25,
        high_threshold=0.5,
        critical_threshold=0.75,
        statistical_significance=0.01,
        minimum_samples=100
    )
    
    drift_monitor = DriftMonitor(custom_thresholds)
    
    print(f"📊 Dataset Information:")
    print(f"   Reference dataset: {len(reference_data)} samples, {len(reference_data.columns)} features")
    print(f"   Features: {', '.join(reference_data.columns)}")
    print()
    
    # Test scenarios
    scenarios = [
        ("No Drift Scenario", "baseline_dataset", no_drift_data),
        ("Moderate Drift Scenario", "moderate_drift_dataset", moderate_drift_data),
        ("High Drift Scenario", "high_drift_dataset", high_drift_data)
    ]
    
    for scenario_name, dataset_id, current_data in scenarios:
        print(f"🧪 {scenario_name}")
        print("-" * 40)
        
        # Monitor drift
        report = drift_monitor.monitor_drift(
            dataset_id=dataset_id,
            reference_dataset_id="reference_dataset",
            current_data=current_data,
            reference_data=reference_data
        )
        
        # Display results
        print(f"   Overall Drift Score: {report.drift_score:.3f}")
        print(f"   Severity Level: {report.get_severity_level().value.upper()}")
        print(f"   Significant Drift: {'Yes' if report.has_significant_drift() else 'No'}")
        print(f"   Generated Alerts: {len(report.alerts)}")
        print(f"   Recommendations: {len(report.recommendations)}")
        
        # Show feature-level drift scores
        print(f"   Feature Drift Scores:")
        for feature, score in sorted(report.feature_drift_scores.items(), key=lambda x: x[1], reverse=True):
            status = "🔴" if score > 0.5 else "🟡" if score > 0.25 else "🟢"
            print(f"     {status} {feature}: {score:.3f}")
        
        # Show statistical tests
        print(f"   Statistical Tests:")
        for feature, test in report.statistical_tests.items():
            significance = "Significant" if test.is_significant else "Not Significant"
            print(f"     {feature}: {test.test_name} (p={test.p_value:.4f}) - {significance}")
        
        # Show alerts
        if report.alerts:
            print(f"   Alerts Generated:")
            for alert in report.alerts:
                print(f"     🚨 {alert.severity.value.upper()}: {alert.message}")
                print(f"        Affected Features: {', '.join(alert.affected_features)}")
        
        # Show recommendations
        if report.recommendations:
            print(f"   Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"     {i}. {rec.description} (Priority: {rec.priority})")
                print(f"        Estimated Effort: {rec.estimated_effort}")
        
        print()


async def demonstrate_alert_management():
    """Demonstrate alert management and notification system."""
    print("🚨 Alert Management and Notification System Demo")
    print("=" * 60)
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Configure notification channels
    print("🔧 Configuring Notification Channels...")
    
    # Email configuration (mock)
    email_config = NotificationConfig(
        channel=NotificationChannel.EMAIL,
        config={
            'smtp_server': 'smtp.company.com',
            'smtp_port': 587,
            'from_email': 'ai-alerts@company.com',
            'to_emails': ['data-team@company.com', 'ml-ops@company.com'],
            'use_tls': True,
            'username': 'ai-alerts@company.com',
            'password': 'secure_password'
        },
        retry_attempts=3,
        retry_delay=30
    )
    
    # Slack configuration (mock)
    slack_config = NotificationConfig(
        channel=NotificationChannel.SLACK,
        config={
            'webhook_url': 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX'
        },
        retry_attempts=2,
        retry_delay=15
    )
    
    # Webhook configuration (mock)
    webhook_config = NotificationConfig(
        channel=NotificationChannel.WEBHOOK,
        config={
            'webhook_url': 'https://api.company.com/alerts/webhook',
            'headers': {'Authorization': 'Bearer token123', 'Content-Type': 'application/json'},
            'custom_fields': {'environment': 'production', 'service': 'ai-data-readiness'}
        }
    )
    
    try:
        alert_manager.configure_notification_channel(NotificationChannel.EMAIL, email_config)
        alert_manager.configure_notification_channel(NotificationChannel.SLACK, slack_config)
        alert_manager.configure_notification_channel(NotificationChannel.WEBHOOK, webhook_config)
        print("   ✅ All notification channels configured successfully")
    except Exception as e:
        print(f"   ⚠️  Configuration note: {str(e)} (using mock configurations)")
    
    # Configure alert rules
    print("\n📋 Setting Up Alert Rules...")
    
    # High drift rule
    high_drift_rule = AlertRule(
        name="high_drift_production",
        description="Alert for high drift in production datasets",
        conditions={
            'dataset_pattern': r'production_.*',
            'drift_score_threshold': 0.4
        },
        severity=AlertSeverity.HIGH,
        notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
        cooldown_period=600,  # 10 minutes
        enabled=True
    )
    
    # Critical drift rule
    critical_drift_rule = AlertRule(
        name="critical_drift_all",
        description="Alert for critical drift in any dataset",
        conditions={
            'min_drift_score': 0.7
        },
        severity=AlertSeverity.CRITICAL,
        notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK],
        cooldown_period=300,  # 5 minutes
        enabled=True
    )
    
    alert_manager.add_alert_rule(high_drift_rule)
    alert_manager.add_alert_rule(critical_drift_rule)
    print("   ✅ Alert rules configured")
    
    # Configure escalation rules
    print("\n⬆️  Setting Up Escalation Rules...")
    
    critical_escalation = EscalationRule(
        severity=AlertSeverity.CRITICAL,
        escalation_delay=15,  # 15 minutes
        escalation_levels=[EscalationLevel.LEVEL_1, EscalationLevel.LEVEL_2],
        notification_channels=[NotificationChannel.EMAIL],
        max_escalations=2
    )
    
    alert_manager.add_escalation_rule(AlertSeverity.CRITICAL, critical_escalation)
    print("   ✅ Escalation rules configured")
    
    # Create sample drift report with alerts
    print("\n🧪 Simulating Drift Detection and Alert Processing...")
    
    # Create sample datasets for alert testing
    reference_data, _, _, high_drift_data = create_sample_datasets()
    
    # Initialize drift monitor
    drift_monitor = DriftMonitor()
    
    # Generate drift report
    drift_report = drift_monitor.monitor_drift(
        dataset_id="production_customer_data",
        reference_dataset_id="baseline_customer_data",
        current_data=high_drift_data,
        reference_data=reference_data
    )
    
    print(f"   📊 Drift Report Generated:")
    print(f"      Dataset: {drift_report.dataset_id}")
    print(f"      Overall Drift Score: {drift_report.drift_score:.3f}")
    print(f"      Alerts Generated: {len(drift_report.alerts)}")
    
    # Process drift report through alert manager
    print("\n📤 Processing Alerts...")
    
    # Mock the notification sending to avoid actual external calls
    original_send_method = alert_manager._send_alert_notifications
    
    async def mock_send_notifications(alert):
        print(f"   📧 [MOCK] Sending alert via configured channels:")
        channels = alert_manager._get_channels_for_alert(alert)
        for channel in channels:
            if channel in alert_manager.notification_configs:
                print(f"      - {channel.value}: Alert '{alert.message}' sent successfully")
        return True
    
    alert_manager._send_alert_notifications = mock_send_notifications
    
    try:
        processed_alerts = await alert_manager.process_drift_report(drift_report)
        print(f"   ✅ Processed {len(processed_alerts)} alerts")
        
        # Show alert details
        for alert in processed_alerts:
            print(f"\n   🚨 Alert Details:")
            print(f"      ID: {alert.id}")
            print(f"      Severity: {alert.severity.value.upper()}")
            print(f"      Message: {alert.message}")
            print(f"      Drift Score: {alert.drift_score:.3f}")
            print(f"      Affected Features: {', '.join(alert.affected_features)}")
            print(f"      Detected At: {alert.detected_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    finally:
        # Restore original method
        alert_manager._send_alert_notifications = original_send_method
    
    # Demonstrate alert management operations
    print("\n🔧 Demonstrating Alert Management Operations...")
    
    if processed_alerts:
        sample_alert = processed_alerts[0]
        
        # Acknowledge alert
        alert_manager.acknowledge_alert(sample_alert.id, "data-engineer@company.com")
        print(f"   ✅ Alert {sample_alert.id} acknowledged")
        
        # Get active alerts
        active_alerts = alert_manager.get_active_alerts()
        print(f"   📋 Active alerts: {len(active_alerts)}")
        
        # Get alert statistics
        stats = alert_manager.get_alert_statistics(hours=24)
        print(f"   📊 Alert Statistics (24h):")
        print(f"      Total alerts: {stats['total_alerts']}")
        print(f"      Active alerts: {stats['active_alerts']}")
        print(f"      Acknowledgment rate: {stats['acknowledgment_rate']:.1%}")
        print(f"      Severity breakdown: {stats['severity_breakdown']}")
        
        # Resolve alert
        alert_manager.resolve_alert(sample_alert.id, "ml-ops-lead@company.com")
        print(f"   ✅ Alert {sample_alert.id} resolved")


async def demonstrate_advanced_features():
    """Demonstrate advanced drift monitoring features."""
    print("\n🚀 Advanced Drift Monitoring Features")
    print("=" * 60)
    
    # Create datasets with specific drift patterns
    np.random.seed(100)
    
    # Dataset with concept drift (target relationship changes)
    print("🎯 Concept Drift Detection:")
    reference_features = pd.DataFrame({
        'feature_a': np.random.normal(0, 1, 2000),
        'feature_b': np.random.normal(0, 1, 2000)
    })
    
    current_features = pd.DataFrame({
        'feature_a': np.random.normal(0, 1, 2000),  # Same distribution
        'feature_b': np.random.normal(0.5, 1.2, 2000)  # Shifted distribution
    })
    
    drift_monitor = DriftMonitor()
    concept_drift_report = drift_monitor.monitor_drift(
        dataset_id="concept_drift_test",
        reference_dataset_id="reference",
        current_data=current_features,
        reference_data=reference_features
    )
    
    print(f"   Overall Drift Score: {concept_drift_report.drift_score:.3f}")
    print(f"   Feature A Drift: {concept_drift_report.feature_drift_scores['feature_a']:.3f}")
    print(f"   Feature B Drift: {concept_drift_report.feature_drift_scores['feature_b']:.3f}")
    
    # Demonstrate drift metrics
    print(f"\n📈 Detailed Drift Metrics:")
    metrics = concept_drift_report.metrics
    if metrics:
        print(f"   Drift Velocity: {metrics.drift_velocity:.3f}")
        print(f"   Drift Magnitude: {metrics.drift_magnitude:.3f}")
        print(f"   Confidence Interval: ({metrics.confidence_interval[0]:.3f}, {metrics.confidence_interval[1]:.3f})")
        
        print(f"   Distribution Distances:")
        for feature, distance in metrics.distribution_distances.items():
            print(f"     {feature}: {distance:.3f}")
    
    # Demonstrate custom thresholds
    print(f"\n⚙️  Custom Threshold Configuration:")
    custom_thresholds = DriftThresholds(
        low_threshold=0.05,
        medium_threshold=0.15,
        high_threshold=0.35,
        critical_threshold=0.6,
        statistical_significance=0.001,
        minimum_samples=200
    )
    
    custom_monitor = DriftMonitor(custom_thresholds)
    print(f"   Configured thresholds: Low={custom_thresholds.low_threshold}, "
          f"Medium={custom_thresholds.medium_threshold}, "
          f"High={custom_thresholds.high_threshold}, "
          f"Critical={custom_thresholds.critical_threshold}")
    
    # Test with custom thresholds
    custom_report = custom_monitor.monitor_drift(
        dataset_id="custom_threshold_test",
        reference_dataset_id="reference",
        current_data=current_features,
        reference_data=reference_features
    )
    
    print(f"   With custom thresholds - Overall Score: {custom_report.drift_score:.3f}")
    print(f"   Severity Level: {custom_report.get_severity_level().value.upper()}")
    print(f"   Alerts Generated: {len(custom_report.alerts)}")


async def main():
    """Main demo function."""
    print("🌟 AI Data Readiness Platform - Drift Monitoring & Alerting Demo")
    print("=" * 80)
    print()
    
    try:
        # Demonstrate drift detection
        await demonstrate_drift_detection()
        
        # Demonstrate alert management
        await demonstrate_alert_management()
        
        # Demonstrate advanced features
        await demonstrate_advanced_features()
        
        print("\n✅ Demo completed successfully!")
        print("\n🎉 Key Features Demonstrated:")
        print("• Statistical drift detection with multiple algorithms (PSI, KS test, Chi-square)")
        print("• Feature-level and overall drift scoring")
        print("• Configurable thresholds and severity levels")
        print("• Comprehensive alert generation and management")
        print("• Multiple notification channels (Email, Slack, Webhook)")
        print("• Alert escalation and acknowledgment workflows")
        print("• Custom alert rules and conditions")
        print("• Alert cooldown and deduplication")
        print("• Statistical test interpretation and recommendations")
        print("• Drift metrics including velocity and confidence intervals")
        print("• Support for both numeric and categorical features")
        
        print(f"\n📊 Final Statistics:")
        print(f"   Drift detection algorithms: 5+ (PSI, KS, Chi-square, JS divergence, etc.)")
        print(f"   Notification channels: 3 (Email, Slack, Webhook)")
        print(f"   Alert severity levels: 4 (Low, Medium, High, Critical)")
        print(f"   Statistical tests: 2+ (KS test, Chi-square)")
        print(f"   Supported data types: Numeric and Categorical")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())