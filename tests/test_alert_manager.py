"""
Tests for alert management and notification system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

from ai_data_readiness.engines.alert_manager import (
    AlertManager, NotificationChannel, NotificationConfig, AlertRule,
    EscalationRule, EscalationLevel, NotificationTemplate,
    EmailNotificationProvider, SlackNotificationProvider, WebhookNotificationProvider
)
from ai_data_readiness.models.drift_models import (
    DriftAlert, AlertSeverity, DriftType, DriftReport, StatisticalTest
)


class TestNotificationProviders:
    """Test cases for notification providers."""
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample drift alert for testing."""
        return DriftAlert(
            id="test_alert_001",
            dataset_id="test_dataset",
            drift_type=DriftType.COVARIATE_SHIFT,
            severity=AlertSeverity.HIGH,
            message="High drift detected in dataset",
            affected_features=["feature1", "feature2"],
            drift_score=0.65,
            threshold=0.5,
            detected_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def email_config(self):
        """Create email notification configuration."""
        return NotificationConfig(
            channel=NotificationChannel.EMAIL,
            config={
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'from_email': 'alerts@example.com',
                'to_emails': ['admin@example.com', 'team@example.com'],
                'use_tls': True,
                'username': 'alerts@example.com',
                'password': 'password123'
            }
        )
    
    @pytest.fixture
    def email_template(self):
        """Create email notification template."""
        return NotificationTemplate(
            channel=NotificationChannel.EMAIL,
            subject_template="[{severity}] Drift Alert - {dataset_id}",
            body_template="Alert: {message}\nScore: {drift_score}\nFeatures: {affected_features}",
            format_type="text"
        )
    
    def test_email_provider_validation(self):
        """Test email provider configuration validation."""
        provider = EmailNotificationProvider()
        
        # Valid configuration
        valid_config = {
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587,
            'from_email': 'test@example.com',
            'to_emails': ['admin@example.com']
        }
        assert provider.validate_config(valid_config)
        
        # Invalid configuration (missing required field)
        invalid_config = {
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587
            # Missing from_email and to_emails
        }
        assert not provider.validate_config(invalid_config)
    
    def test_email_template_formatting(self, sample_alert):
        """Test email template formatting."""
        provider = EmailNotificationProvider()
        template = "Alert {alert_id}: {message} (Score: {drift_score})"
        
        formatted = provider._format_template(template, sample_alert)
        
        assert "test_alert_001" in formatted
        assert "High drift detected" in formatted
        assert "0.65" in formatted
    
    @pytest.mark.asyncio
    async def test_email_provider_send_notification_mock(self, sample_alert, email_config, email_template):
        """Test email provider send notification with mocking."""
        provider = EmailNotificationProvider()
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await provider.send_notification(sample_alert, email_config, email_template)
            
            assert result is True
            mock_server.send_message.assert_called_once()
    
    def test_slack_provider_validation(self):
        """Test Slack provider configuration validation."""
        provider = SlackNotificationProvider()
        
        # Valid configuration
        valid_config = {'webhook_url': 'https://hooks.slack.com/services/xxx'}
        assert provider.validate_config(valid_config)
        
        # Invalid configuration
        invalid_config = {}
        assert not provider.validate_config(invalid_config)
    
    def test_slack_color_mapping(self):
        """Test Slack color mapping for severity levels."""
        provider = SlackNotificationProvider()
        
        assert provider._get_color_for_severity(AlertSeverity.LOW) == "good"
        assert provider._get_color_for_severity(AlertSeverity.MEDIUM) == "warning"
        assert provider._get_color_for_severity(AlertSeverity.HIGH) == "danger"
        assert provider._get_color_for_severity(AlertSeverity.CRITICAL) == "#ff0000"
    
    @pytest.mark.asyncio
    async def test_slack_provider_send_notification_mock(self, sample_alert):
        """Test Slack provider send notification with mocking."""
        provider = SlackNotificationProvider()
        config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            config={'webhook_url': 'https://hooks.slack.com/services/test'}
        )
        template = NotificationTemplate(
            channel=NotificationChannel.SLACK,
            subject_template="Alert: {dataset_id}",
            body_template="Drift detected",
            format_type="text"
        )
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            result = await provider.send_notification(sample_alert, config, template)
            
            assert result is True
            mock_post.assert_called_once()
            
            # Check payload structure
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert 'text' in payload
            assert 'attachments' in payload
            assert len(payload['attachments']) == 1
    
    def test_webhook_provider_validation(self):
        """Test webhook provider configuration validation."""
        provider = WebhookNotificationProvider()
        
        # Valid configuration
        valid_config = {'webhook_url': 'https://api.example.com/webhook'}
        assert provider.validate_config(valid_config)
        
        # Invalid configuration
        invalid_config = {}
        assert not provider.validate_config(invalid_config)
    
    @pytest.mark.asyncio
    async def test_webhook_provider_send_notification_mock(self, sample_alert):
        """Test webhook provider send notification with mocking."""
        provider = WebhookNotificationProvider()
        config = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            config={
                'webhook_url': 'https://api.example.com/webhook',
                'headers': {'Authorization': 'Bearer token123'},
                'custom_fields': {'environment': 'production'}
            }
        )
        template = NotificationTemplate(
            channel=NotificationChannel.WEBHOOK,
            subject_template="Webhook Alert",
            body_template="Alert data",
            format_type="json"
        )
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            result = await provider.send_notification(sample_alert, config, template)
            
            assert result is True
            mock_post.assert_called_once()
            
            # Check payload structure
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert payload['alert_id'] == sample_alert.id
            assert payload['dataset_id'] == sample_alert.dataset_id
            assert payload['severity'] == sample_alert.severity.value
            assert payload['environment'] == 'production'  # Custom field


class TestAlertManager:
    """Test cases for AlertManager class."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create AlertManager instance for testing."""
        return AlertManager()
    
    @pytest.fixture
    def sample_drift_report(self):
        """Create sample drift report for testing."""
        alert = DriftAlert(
            id="report_alert_001",
            dataset_id="test_dataset",
            drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.MEDIUM,
            message="Feature drift detected",
            affected_features=["feature1"],
            drift_score=0.35,
            threshold=0.3
        )
        
        return DriftReport(
            dataset_id="test_dataset",
            reference_dataset_id="reference_dataset",
            drift_score=0.35,
            feature_drift_scores={"feature1": 0.35, "feature2": 0.1},
            statistical_tests={
                "feature1": StatisticalTest(
                    test_name="KS Test",
                    statistic=0.2,
                    p_value=0.01,
                    threshold=0.05,
                    is_significant=True,
                    interpretation="Significant drift detected"
                )
            },
            alerts=[alert]
        )
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test AlertManager initialization."""
        assert len(alert_manager.notification_providers) == 3
        assert NotificationChannel.EMAIL in alert_manager.notification_providers
        assert NotificationChannel.SLACK in alert_manager.notification_providers
        assert NotificationChannel.WEBHOOK in alert_manager.notification_providers
        
        # Check default templates are initialized
        assert len(alert_manager.notification_templates) == 3
    
    def test_configure_notification_channel(self, alert_manager):
        """Test notification channel configuration."""
        config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            config={
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'from_email': 'test@example.com',
                'to_emails': ['admin@example.com']
            }
        )
        
        alert_manager.configure_notification_channel(NotificationChannel.EMAIL, config)
        
        assert NotificationChannel.EMAIL in alert_manager.notification_configs
        assert alert_manager.notification_configs[NotificationChannel.EMAIL] == config
    
    def test_configure_invalid_notification_channel(self, alert_manager):
        """Test configuration with invalid channel."""
        config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            config={'invalid': 'config'}  # Missing required fields
        )
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            alert_manager.configure_notification_channel(NotificationChannel.EMAIL, config)
    
    def test_add_alert_rule(self, alert_manager):
        """Test adding alert rules."""
        rule = AlertRule(
            name="high_drift_rule",
            description="Alert for high drift",
            conditions={'min_drift_score': 0.5},
            severity=AlertSeverity.HIGH,
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
        )
        
        alert_manager.add_alert_rule(rule)
        
        assert len(alert_manager.alert_rules) == 1
        assert alert_manager.alert_rules[0] == rule
    
    def test_add_escalation_rule(self, alert_manager):
        """Test adding escalation rules."""
        escalation_rule = EscalationRule(
            severity=AlertSeverity.CRITICAL,
            escalation_delay=15,  # 15 minutes
            escalation_levels=[EscalationLevel.LEVEL_1, EscalationLevel.LEVEL_2],
            notification_channels=[NotificationChannel.EMAIL],
            max_escalations=2
        )
        
        alert_manager.add_escalation_rule(AlertSeverity.CRITICAL, escalation_rule)
        
        assert AlertSeverity.CRITICAL in alert_manager.escalation_rules
        assert alert_manager.escalation_rules[AlertSeverity.CRITICAL] == escalation_rule
    
    def test_set_notification_template(self, alert_manager):
        """Test setting notification templates."""
        template = NotificationTemplate(
            channel=NotificationChannel.EMAIL,
            subject_template="Custom Alert: {dataset_id}",
            body_template="Custom message: {message}",
            format_type="html"
        )
        
        alert_manager.set_notification_template(NotificationChannel.EMAIL, template)
        
        assert alert_manager.notification_templates[NotificationChannel.EMAIL] == template
    
    @pytest.mark.asyncio
    async def test_process_drift_report(self, alert_manager, sample_drift_report):
        """Test processing drift report."""
        # Mock notification sending
        with patch.object(alert_manager, '_send_alert_notifications', return_value=True) as mock_send:
            generated_alerts = await alert_manager.process_drift_report(sample_drift_report)
            
            assert len(generated_alerts) == 1
            assert generated_alerts[0].id == "report_alert_001"
            mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_alert(self, alert_manager):
        """Test sending individual alert."""
        alert = DriftAlert(
            id="test_alert_002",
            dataset_id="test_dataset",
            drift_type=DriftType.COVARIATE_SHIFT,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            affected_features=["feature1"],
            drift_score=0.6,
            threshold=0.5
        )
        
        with patch.object(alert_manager, '_send_alert_notifications', return_value=True) as mock_send:
            result = await alert_manager.send_alert(alert)
            
            assert result is True
            mock_send.assert_called_once_with(alert)
            assert alert.id in alert_manager.active_alerts
    
    def test_acknowledge_alert(self, alert_manager):
        """Test alert acknowledgment."""
        alert = DriftAlert(
            id="test_alert_003",
            dataset_id="test_dataset",
            drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.MEDIUM,
            message="Test alert",
            affected_features=["feature1"],
            drift_score=0.4,
            threshold=0.3
        )
        
        # Add alert to active alerts
        alert_manager.active_alerts[alert.id] = alert
        
        result = alert_manager.acknowledge_alert(alert.id, "admin@example.com")
        
        assert result is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin@example.com"
        assert alert.acknowledged_at is not None
    
    def test_resolve_alert(self, alert_manager):
        """Test alert resolution."""
        alert = DriftAlert(
            id="test_alert_004",
            dataset_id="test_dataset",
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=AlertSeverity.LOW,
            message="Test alert",
            affected_features=["feature1"],
            drift_score=0.2,
            threshold=0.1
        )
        
        # Add alert to active alerts
        alert_manager.active_alerts[alert.id] = alert
        
        result = alert_manager.resolve_alert(alert.id, "admin@example.com")
        
        assert result is True
        assert alert.id not in alert_manager.active_alerts
        assert len(alert_manager.alert_history) == 1
        assert alert_manager.alert_history[0].acknowledged is True
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        # Add some test alerts
        alert1 = DriftAlert(
            id="alert_001", dataset_id="dataset1", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.HIGH, message="Alert 1", affected_features=["f1"],
            drift_score=0.6, threshold=0.5
        )
        alert2 = DriftAlert(
            id="alert_002", dataset_id="dataset2", drift_type=DriftType.COVARIATE_SHIFT,
            severity=AlertSeverity.MEDIUM, message="Alert 2", affected_features=["f2"],
            drift_score=0.4, threshold=0.3
        )
        
        alert_manager.active_alerts[alert1.id] = alert1
        alert_manager.active_alerts[alert2.id] = alert2
        
        # Get all active alerts
        all_alerts = alert_manager.get_active_alerts()
        assert len(all_alerts) == 2
        
        # Get filtered alerts
        high_alerts = alert_manager.get_active_alerts(AlertSeverity.HIGH)
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == AlertSeverity.HIGH
    
    def test_get_alert_history(self, alert_manager):
        """Test getting alert history."""
        # Add some historical alerts
        old_alert = DriftAlert(
            id="old_alert", dataset_id="dataset1", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.LOW, message="Old alert", affected_features=["f1"],
            drift_score=0.2, threshold=0.1,
            detected_at=datetime.utcnow() - timedelta(hours=48)  # 2 days ago
        )
        recent_alert = DriftAlert(
            id="recent_alert", dataset_id="dataset2", drift_type=DriftType.COVARIATE_SHIFT,
            severity=AlertSeverity.MEDIUM, message="Recent alert", affected_features=["f2"],
            drift_score=0.4, threshold=0.3,
            detected_at=datetime.utcnow() - timedelta(hours=12)  # 12 hours ago
        )
        
        alert_manager.alert_history.extend([old_alert, recent_alert])
        
        # Get last 24 hours
        recent_history = alert_manager.get_alert_history(hours=24)
        assert len(recent_history) == 1
        assert recent_history[0].id == "recent_alert"
        
        # Get last 72 hours
        extended_history = alert_manager.get_alert_history(hours=72)
        assert len(extended_history) == 2
    
    def test_get_alert_statistics(self, alert_manager):
        """Test getting alert statistics."""
        # Add some test data
        alert1 = DriftAlert(
            id="stat_alert_1", dataset_id="dataset1", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.HIGH, message="Alert 1", affected_features=["f1"],
            drift_score=0.6, threshold=0.5, acknowledged=True
        )
        alert2 = DriftAlert(
            id="stat_alert_2", dataset_id="dataset2", drift_type=DriftType.COVARIATE_SHIFT,
            severity=AlertSeverity.MEDIUM, message="Alert 2", affected_features=["f2"],
            drift_score=0.4, threshold=0.3, acknowledged=False
        )
        
        alert_manager.active_alerts[alert1.id] = alert1
        alert_manager.alert_history.append(alert2)
        
        stats = alert_manager.get_alert_statistics(hours=24)
        
        assert stats['total_alerts'] == 2
        assert stats['active_alerts'] == 1
        assert stats['resolved_alerts'] == 1
        assert stats['severity_breakdown']['high'] == 1
        assert stats['severity_breakdown']['medium'] == 1
        assert stats['acknowledgment_rate'] == 0.5  # 1 out of 2 acknowledged
    
    def test_should_send_alert_cooldown(self, alert_manager):
        """Test alert cooldown logic."""
        alert = DriftAlert(
            id="cooldown_alert", dataset_id="test_dataset", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.MEDIUM, message="Test alert", affected_features=["f1"],
            drift_score=0.4, threshold=0.3
        )
        
        # First alert should be sent
        assert alert_manager._should_send_alert(alert) is True
        
        # Set cooldown
        cooldown_key = f"{alert.dataset_id}_{alert.drift_type.value}_{alert.severity.value}"
        alert_manager.alert_cooldowns[cooldown_key] = datetime.utcnow() + timedelta(minutes=10)
        
        # Second alert should be blocked by cooldown
        assert alert_manager._should_send_alert(alert) is False
    
    def test_get_channels_for_alert(self, alert_manager):
        """Test channel selection for alerts."""
        # Test default channel selection
        critical_alert = DriftAlert(
            id="critical", dataset_id="test", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.CRITICAL, message="Critical", affected_features=["f1"],
            drift_score=0.8, threshold=0.7
        )
        
        channels = alert_manager._get_channels_for_alert(critical_alert)
        assert NotificationChannel.EMAIL in channels
        assert NotificationChannel.SLACK in channels
        assert NotificationChannel.WEBHOOK in channels
        
        # Test with custom rule
        rule = AlertRule(
            name="custom_rule",
            description="Custom rule",
            conditions={},
            severity=AlertSeverity.CRITICAL,
            notification_channels=[NotificationChannel.EMAIL]
        )
        alert_manager.add_alert_rule(rule)
        
        channels_custom = alert_manager._get_channels_for_alert(critical_alert)
        assert channels_custom == [NotificationChannel.EMAIL]
    
    def test_alert_matches_rule(self, alert_manager):
        """Test alert rule matching."""
        alert = DriftAlert(
            id="match_test", dataset_id="production_dataset", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.HIGH, message="Test", affected_features=["f1"],
            drift_score=0.7, threshold=0.5
        )
        
        # Rule that should match
        matching_rule = AlertRule(
            name="production_rule",
            description="Production alerts",
            conditions={
                'dataset_pattern': r'production_.*',
                'drift_score_threshold': 0.6
            },
            severity=AlertSeverity.HIGH,
            notification_channels=[NotificationChannel.EMAIL]
        )
        
        assert alert_manager._alert_matches_rule(alert, matching_rule) is True
        
        # Rule that shouldn't match (different severity)
        non_matching_rule = AlertRule(
            name="low_severity_rule",
            description="Low severity alerts",
            conditions={},
            severity=AlertSeverity.LOW,
            notification_channels=[NotificationChannel.WEBHOOK]
        )
        
        assert alert_manager._alert_matches_rule(alert, non_matching_rule) is False
    
    def test_evaluate_custom_rules(self, alert_manager, sample_drift_report):
        """Test custom rule evaluation."""
        # Add custom rule
        rule = AlertRule(
            name="high_drift_custom",
            description="Custom high drift rule",
            conditions={'min_drift_score': 0.3},
            severity=AlertSeverity.HIGH,
            notification_channels=[NotificationChannel.EMAIL],
            enabled=True
        )
        alert_manager.add_alert_rule(rule)
        
        custom_alerts = alert_manager._evaluate_custom_rules(sample_drift_report)
        
        assert len(custom_alerts) == 1
        assert custom_alerts[0].severity == AlertSeverity.HIGH
        assert "high_drift_custom" in custom_alerts[0].id
    
    @pytest.mark.asyncio
    async def test_send_alert_notifications_mock(self, alert_manager):
        """Test sending alert notifications with mocking."""
        alert = DriftAlert(
            id="notification_test", dataset_id="test", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.MEDIUM, message="Test", affected_features=["f1"],
            drift_score=0.4, threshold=0.3
        )
        
        # Configure Slack channel (which is used for MEDIUM severity by default)
        config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            config={'webhook_url': 'https://hooks.slack.com/services/test'}
        )
        alert_manager.configure_notification_channel(NotificationChannel.SLACK, config)
        
        # Mock the provider
        with patch.object(alert_manager.notification_providers[NotificationChannel.SLACK], 
                         'send_notification', return_value=True) as mock_send:
            
            result = await alert_manager._send_alert_notifications(alert)
            
            assert result is True
            mock_send.assert_called_once()
    
    def test_cooldown_period_calculation(self, alert_manager):
        """Test cooldown period calculation."""
        # Test default cooldowns
        critical_alert = DriftAlert(
            id="critical", dataset_id="test", drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.CRITICAL, message="Critical", affected_features=["f1"],
            drift_score=0.8, threshold=0.7
        )
        
        cooldown = alert_manager._get_cooldown_period(critical_alert)
        assert cooldown == 300  # 5 minutes for critical
        
        # Test custom rule cooldown
        rule = AlertRule(
            name="custom_cooldown",
            description="Custom cooldown rule",
            conditions={},
            severity=AlertSeverity.CRITICAL,
            notification_channels=[NotificationChannel.EMAIL],
            cooldown_period=600  # 10 minutes
        )
        alert_manager.add_alert_rule(rule)
        
        custom_cooldown = alert_manager._get_cooldown_period(critical_alert)
        assert custom_cooldown == 600


if __name__ == "__main__":
    pytest.main([__file__])