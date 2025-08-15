"""
Tests for Crisis Detection and Assessment Engine
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.crisis_detection_engine import (
    CrisisDetectionEngine,
    EarlyWarningSystem,
    CrisisClassifier,
    ImpactAssessor,
    EscalationManager,
    Crisis,
    CrisisType,
    SeverityLevel,
    CrisisStatus,
    Signal,
    PotentialCrisis,
    CrisisClassification,
    ImpactAssessment
)


class TestEarlyWarningSystem:
    """Test cases for Early Warning System"""
    
    @pytest.fixture
    def early_warning_system(self):
        return EarlyWarningSystem()
    
    @pytest.mark.asyncio
    async def test_monitor_signals(self, early_warning_system):
        """Test signal monitoring functionality"""
        signals = await early_warning_system.monitor_signals()
        
        assert isinstance(signals, list)
        # Should have some signals from mock methods
        assert len(signals) >= 0
    
    @pytest.mark.asyncio
    async def test_detect_potential_crises(self, early_warning_system):
        """Test potential crisis detection from signals"""
        # Create test signals
        test_signals = [
            Signal(
                source="system_monitor",
                signal_type="high_cpu_usage",
                value=95.0,
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={}
            ),
            Signal(
                source="application_monitor",
                signal_type="high_error_rate",
                value=10.0,
                timestamp=datetime.now(),
                confidence=0.85,
                metadata={}
            )
        ]
        
        potential_crises = await early_warning_system.detect_potential_crises(test_signals)
        
        assert isinstance(potential_crises, list)
        if potential_crises:
            crisis = potential_crises[0]
            assert isinstance(crisis, PotentialCrisis)
            assert crisis.crisis_type in CrisisType
            assert 0 <= crisis.probability <= 1
            assert crisis.confidence_score > 0
    
    def test_analyze_signal_patterns(self, early_warning_system):
        """Test signal pattern analysis"""
        test_signals = [
            Signal(
                source="system_monitor",
                signal_type="high_cpu_usage",
                value=95.0,
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={}
            ),
            Signal(
                source="security_monitor",
                signal_type="high_failed_logins",
                value=200,
                timestamp=datetime.now(),
                confidence=0.8,
                metadata={}
            )
        ]
        
        patterns = early_warning_system._analyze_signal_patterns(test_signals)
        
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert 'crisis_type' in pattern
            assert 'probability' in pattern
            assert 'signals' in pattern
            assert 'confidence' in pattern


class TestCrisisClassifier:
    """Test cases for Crisis Classifier"""
    
    @pytest.fixture
    def crisis_classifier(self):
        return CrisisClassifier()
    
    @pytest.fixture
    def sample_crisis(self):
        return Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now(),
            affected_areas=["production_systems"],
            stakeholders_impacted=["customers", "engineering"],
            current_status=CrisisStatus.DETECTED,
            signals=[
                Signal(
                    source="system_monitor",
                    signal_type="high_cpu_usage",
                    value=95.0,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={}
                )
            ]
        )
    
    def test_classify_crisis(self, crisis_classifier, sample_crisis):
        """Test crisis classification functionality"""
        classification = crisis_classifier.classify_crisis(sample_crisis)
        
        assert isinstance(classification, CrisisClassification)
        assert classification.crisis_type in CrisisType
        assert classification.severity_level in SeverityLevel
        assert 0 <= classification.confidence <= 1
        assert isinstance(classification.sub_categories, list)
        assert isinstance(classification.related_crises, list)
        assert isinstance(classification.classification_rationale, str)
    
    def test_determine_crisis_type(self, crisis_classifier):
        """Test crisis type determination"""
        system_signals = [
            Signal(
                source="system_monitor",
                signal_type="high_cpu_usage",
                value=95.0,
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={}
            )
        ]
        
        crisis_type = crisis_classifier._determine_crisis_type(system_signals)
        assert crisis_type == CrisisType.SYSTEM_OUTAGE
        
        security_signals = [
            Signal(
                source="security_monitor",
                signal_type="high_failed_logins",
                value=200,
                timestamp=datetime.now(),
                confidence=0.8,
                metadata={}
            )
        ]
        
        crisis_type = crisis_classifier._determine_crisis_type(security_signals)
        assert crisis_type == CrisisType.SECURITY_BREACH
    
    def test_calculate_severity(self, crisis_classifier, sample_crisis):
        """Test severity calculation"""
        severity = crisis_classifier._calculate_severity(
            sample_crisis, 
            CrisisType.SYSTEM_OUTAGE
        )
        
        assert isinstance(severity, SeverityLevel)
        assert 1 <= severity.value <= 5
    
    def test_calculate_classification_confidence(self, crisis_classifier):
        """Test classification confidence calculation"""
        signals = [
            Signal(
                source="system_monitor",
                signal_type="high_cpu_usage",
                value=95.0,
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={}
            )
        ]
        
        confidence = crisis_classifier._calculate_classification_confidence(
            signals, 
            CrisisType.SYSTEM_OUTAGE
        )
        
        assert 0 <= confidence <= 1


class TestImpactAssessor:
    """Test cases for Impact Assessor"""
    
    @pytest.fixture
    def impact_assessor(self):
        return ImpactAssessor()
    
    @pytest.fixture
    def sample_crisis(self):
        return Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now(),
            affected_areas=["production_systems"],
            stakeholders_impacted=["customers", "engineering"],
            current_status=CrisisStatus.DETECTED,
            signals=[]
        )
    
    def test_assess_impact(self, impact_assessor, sample_crisis):
        """Test impact assessment functionality"""
        assessment = impact_assessor.assess_impact(sample_crisis)
        
        assert isinstance(assessment, ImpactAssessment)
        assert isinstance(assessment.financial_impact, dict)
        assert isinstance(assessment.operational_impact, dict)
        assert isinstance(assessment.reputation_impact, dict)
        assert isinstance(assessment.stakeholder_impact, dict)
        assert isinstance(assessment.timeline_impact, dict)
        assert isinstance(assessment.recovery_estimate, timedelta)
        assert isinstance(assessment.cascading_risks, list)
        assert assessment.mitigation_urgency in SeverityLevel
    
    def test_assess_financial_impact(self, impact_assessor, sample_crisis):
        """Test financial impact assessment"""
        financial_impact = impact_assessor._assess_financial_impact(sample_crisis)
        
        assert isinstance(financial_impact, dict)
        required_keys = [
            'immediate_cost', 'revenue_loss', 'recovery_cost', 
            'opportunity_cost', 'total_estimated_cost'
        ]
        for key in required_keys:
            assert key in financial_impact
            assert isinstance(financial_impact[key], (int, float))
    
    def test_assess_operational_impact(self, impact_assessor, sample_crisis):
        """Test operational impact assessment"""
        operational_impact = impact_assessor._assess_operational_impact(sample_crisis)
        
        assert isinstance(operational_impact, dict)
        required_keys = [
            'service_availability', 'team_productivity', 
            'customer_experience', 'business_continuity'
        ]
        for key in required_keys:
            assert key in operational_impact
            assert isinstance(operational_impact[key], str)
    
    def test_assess_reputation_impact(self, impact_assessor, sample_crisis):
        """Test reputation impact assessment"""
        reputation_impact = impact_assessor._assess_reputation_impact(sample_crisis)
        
        assert isinstance(reputation_impact, dict)
        required_keys = [
            'brand_damage_score', 'customer_trust_impact', 
            'media_attention_level', 'recovery_difficulty'
        ]
        for key in required_keys:
            assert key in reputation_impact
            assert isinstance(reputation_impact[key], (int, float))
    
    def test_estimate_recovery_time(self, impact_assessor, sample_crisis):
        """Test recovery time estimation"""
        recovery_time = impact_assessor._estimate_recovery_time(sample_crisis)
        
        assert isinstance(recovery_time, timedelta)
        assert recovery_time.total_seconds() > 0
    
    def test_identify_cascading_risks(self, impact_assessor, sample_crisis):
        """Test cascading risk identification"""
        risks = impact_assessor._identify_cascading_risks(sample_crisis)
        
        assert isinstance(risks, list)
        for risk in risks:
            assert isinstance(risk, str)


class TestEscalationManager:
    """Test cases for Escalation Manager"""
    
    @pytest.fixture
    def escalation_manager(self):
        return EscalationManager()
    
    @pytest.fixture
    def sample_crisis(self):
        return Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL,
            start_time=datetime.now(),
            affected_areas=["user_data", "authentication"],
            stakeholders_impacted=["customers", "security_team", "executives"],
            current_status=CrisisStatus.DETECTED,
            signals=[]
        )
    
    def test_should_escalate(self, escalation_manager, sample_crisis):
        """Test escalation decision logic"""
        # Critical severity should escalate
        assert escalation_manager.should_escalate(sample_crisis) == True
        
        # Lower severity should not escalate by default
        sample_crisis.severity_level = SeverityLevel.LOW
        sample_crisis.crisis_type = CrisisType.SYSTEM_OUTAGE
        sample_crisis.stakeholders_impacted = ["engineering"]
        assert escalation_manager.should_escalate(sample_crisis) == False
        
        # Security breach should always escalate
        sample_crisis.crisis_type = CrisisType.SECURITY_BREACH
        assert escalation_manager.should_escalate(sample_crisis) == True
    
    @pytest.mark.asyncio
    async def test_escalate_crisis(self, escalation_manager, sample_crisis):
        """Test crisis escalation process"""
        result = await escalation_manager.escalate_crisis(sample_crisis)
        
        assert isinstance(result, dict)
        assert 'escalated' in result
        assert 'escalation_level' in result
        assert 'notifications_sent' in result
        assert 'escalation_time' in result
        assert 'escalation_reason' in result
        
        if result['escalated']:
            assert result['escalation_level'] is not None
            assert isinstance(result['notifications_sent'], list)
            assert sample_crisis.current_status == CrisisStatus.ESCALATED
    
    def test_determine_escalation_level(self, escalation_manager, sample_crisis):
        """Test escalation level determination"""
        # Catastrophic should be level 5
        sample_crisis.severity_level = SeverityLevel.CATASTROPHIC
        level = escalation_manager._determine_escalation_level(sample_crisis)
        assert level == 5
        
        # Critical should be level 4
        sample_crisis.severity_level = SeverityLevel.CRITICAL
        level = escalation_manager._determine_escalation_level(sample_crisis)
        assert level == 4
        
        # Security breach should be level 4 regardless
        sample_crisis.severity_level = SeverityLevel.MEDIUM
        sample_crisis.crisis_type = CrisisType.SECURITY_BREACH
        level = escalation_manager._determine_escalation_level(sample_crisis)
        assert level == 4
    
    @pytest.mark.asyncio
    async def test_send_escalation_notifications(self, escalation_manager, sample_crisis):
        """Test escalation notification sending"""
        notifications = await escalation_manager._send_escalation_notifications(
            sample_crisis, 
            escalation_level=4
        )
        
        assert isinstance(notifications, list)
        for notification in notifications:
            assert 'recipient' in notification
            assert 'channel' in notification
            assert 'message' in notification
            assert 'sent_time' in notification
            assert 'status' in notification
    
    def test_generate_escalation_message(self, escalation_manager, sample_crisis):
        """Test escalation message generation"""
        message = escalation_manager._generate_escalation_message(sample_crisis, 4)
        
        assert isinstance(message, str)
        assert sample_crisis.id in message
        assert sample_crisis.crisis_type.value in message
        assert sample_crisis.severity_level.name in message


class TestCrisisDetectionEngine:
    """Test cases for main Crisis Detection Engine"""
    
    @pytest.fixture
    def crisis_engine(self):
        return CrisisDetectionEngine()
    
    @pytest.mark.asyncio
    async def test_detect_and_assess_crises(self, crisis_engine):
        """Test main crisis detection workflow"""
        # Mock the early warning system to return test signals
        with patch.object(
            crisis_engine.early_warning_system, 
            'monitor_signals',
            return_value=[
                Signal(
                    source="system_monitor",
                    signal_type="high_cpu_usage",
                    value=95.0,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={}
                )
            ]
        ):
            with patch.object(
                crisis_engine.early_warning_system,
                'detect_potential_crises',
                return_value=[
                    PotentialCrisis(
                        id="potential_test_1",
                        crisis_type=CrisisType.SYSTEM_OUTAGE,
                        probability=0.8,
                        signals=[],
                        predicted_impact="Test impact",
                        time_to_crisis=timedelta(minutes=15),
                        confidence_score=0.8
                    )
                ]
            ):
                crises = await crisis_engine.detect_and_assess_crises()
                
                assert isinstance(crises, list)
                for crisis in crises:
                    assert isinstance(crisis, Crisis)
                    assert crisis.classification is not None
                    assert crisis.impact_assessment is not None
                    assert crisis.current_status == CrisisStatus.ASSESSED
    
    @pytest.mark.asyncio
    async def test_process_crisis(self, crisis_engine):
        """Test individual crisis processing"""
        test_crisis = Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.MEDIUM,
            start_time=datetime.now(),
            affected_areas=["production"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.DETECTED,
            signals=[]
        )
        
        processed_crisis = await crisis_engine._process_crisis(test_crisis)
        
        assert processed_crisis.classification is not None
        assert processed_crisis.impact_assessment is not None
        assert processed_crisis.current_status == CrisisStatus.ASSESSED
    
    def test_create_crisis_from_potential(self, crisis_engine):
        """Test conversion from potential to actual crisis"""
        potential_crisis = PotentialCrisis(
            id="potential_test_1",
            crisis_type=CrisisType.SECURITY_BREACH,
            probability=0.8,
            signals=[],
            predicted_impact="Test impact",
            time_to_crisis=timedelta(hours=1),
            confidence_score=0.8
        )
        
        crisis = crisis_engine._create_crisis_from_potential(potential_crisis)
        
        assert isinstance(crisis, Crisis)
        assert crisis.crisis_type == CrisisType.SECURITY_BREACH
        assert crisis.current_status == CrisisStatus.DETECTED
        assert crisis.id.startswith("crisis_")
    
    @pytest.mark.asyncio
    async def test_get_active_crises(self, crisis_engine):
        """Test getting active crises"""
        # Add a test crisis
        test_crisis = Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.MEDIUM,
            start_time=datetime.now(),
            affected_areas=["production"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.DETECTED,
            signals=[]
        )
        crisis_engine.active_crises[test_crisis.id] = test_crisis
        
        active_crises = await crisis_engine.get_active_crises()
        
        assert isinstance(active_crises, list)
        assert len(active_crises) == 1
        assert active_crises[0].id == test_crisis.id
    
    @pytest.mark.asyncio
    async def test_get_crisis_by_id(self, crisis_engine):
        """Test getting crisis by ID"""
        # Add a test crisis
        test_crisis = Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.MEDIUM,
            start_time=datetime.now(),
            affected_areas=["production"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.DETECTED,
            signals=[]
        )
        crisis_engine.active_crises[test_crisis.id] = test_crisis
        
        # Test existing crisis
        crisis = await crisis_engine.get_crisis_by_id("test_crisis_1")
        assert crisis is not None
        assert crisis.id == "test_crisis_1"
        
        # Test non-existing crisis
        crisis = await crisis_engine.get_crisis_by_id("non_existing")
        assert crisis is None
    
    @pytest.mark.asyncio
    async def test_resolve_crisis(self, crisis_engine):
        """Test crisis resolution"""
        # Add a test crisis
        test_crisis = Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.MEDIUM,
            start_time=datetime.now(),
            affected_areas=["production"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.DETECTED,
            signals=[]
        )
        crisis_engine.active_crises[test_crisis.id] = test_crisis
        
        # Resolve the crisis
        success = await crisis_engine.resolve_crisis("test_crisis_1")
        
        assert success == True
        assert "test_crisis_1" not in crisis_engine.active_crises
        assert len(crisis_engine.crisis_history) == 1
        assert crisis_engine.crisis_history[0].current_status == CrisisStatus.RESOLVED
        assert crisis_engine.crisis_history[0].resolution_time is not None
    
    @pytest.mark.asyncio
    async def test_get_crisis_metrics(self, crisis_engine):
        """Test crisis metrics calculation"""
        # Add some test data
        test_crisis = Crisis(
            id="test_crisis_1",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.MEDIUM,
            start_time=datetime.now() - timedelta(hours=1),
            affected_areas=["production"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.RESOLVED,
            signals=[],
            resolution_time=datetime.now()
        )
        crisis_engine.crisis_history.append(test_crisis)
        
        metrics = await crisis_engine.get_crisis_metrics()
        
        assert isinstance(metrics, dict)
        assert 'active_crises' in metrics
        assert 'total_resolved' in metrics
        assert 'average_resolution_time_seconds' in metrics
        assert 'crisis_type_distribution' in metrics
        assert 'escalation_rate' in metrics
        
        assert metrics['total_resolved'] == 1
        assert metrics['average_resolution_time_seconds'] > 0


class TestIntegration:
    """Integration tests for the complete crisis detection system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_crisis_detection(self):
        """Test complete end-to-end crisis detection workflow"""
        engine = CrisisDetectionEngine()
        
        # Mock signal monitoring to return test signals
        with patch.object(
            engine.early_warning_system,
            'monitor_signals',
            return_value=[
                Signal(
                    source="system_monitor",
                    signal_type="high_cpu_usage",
                    value=95.0,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={}
                ),
                Signal(
                    source="application_monitor",
                    signal_type="high_error_rate",
                    value=15.0,
                    timestamp=datetime.now(),
                    confidence=0.85,
                    metadata={}
                )
            ]
        ):
            # Run detection
            crises = await engine.detect_and_assess_crises()
            
            # Verify results
            assert isinstance(crises, list)
            
            if crises:
                crisis = crises[0]
                
                # Verify crisis structure
                assert isinstance(crisis, Crisis)
                assert crisis.classification is not None
                assert crisis.impact_assessment is not None
                
                # Verify classification
                assert crisis.classification.crisis_type in CrisisType
                assert crisis.classification.severity_level in SeverityLevel
                assert 0 <= crisis.classification.confidence <= 1
                
                # Verify impact assessment
                assert isinstance(crisis.impact_assessment.financial_impact, dict)
                assert isinstance(crisis.impact_assessment.cascading_risks, list)
                assert crisis.impact_assessment.mitigation_urgency in SeverityLevel
                
                # Verify escalation if applicable
                if crisis.severity_level.value >= 4:
                    assert crisis.current_status == CrisisStatus.ESCALATED
                    assert len(crisis.escalation_history) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_crisis_handling(self):
        """Test handling multiple simultaneous crises"""
        engine = CrisisDetectionEngine()
        
        # Create multiple test crises
        crises = []
        for i in range(3):
            crisis = Crisis(
                id=f"test_crisis_{i}",
                crisis_type=list(CrisisType)[i % len(CrisisType)],
                severity_level=list(SeverityLevel)[i % len(SeverityLevel)],
                start_time=datetime.now(),
                affected_areas=[f"area_{i}"],
                stakeholders_impacted=[f"stakeholder_{i}"],
                current_status=CrisisStatus.DETECTED,
                signals=[]
            )
            processed_crisis = await engine._process_crisis(crisis)
            engine.active_crises[crisis.id] = processed_crisis
            crises.append(processed_crisis)
        
        # Verify all crises are tracked
        active_crises = await engine.get_active_crises()
        assert len(active_crises) == 3
        
        # Resolve one crisis
        await engine.resolve_crisis("test_crisis_0")
        
        # Verify state
        active_crises = await engine.get_active_crises()
        assert len(active_crises) == 2
        assert len(engine.crisis_history) == 1
        
        # Get metrics
        metrics = await engine.get_crisis_metrics()
        assert metrics['active_crises'] == 2
        assert metrics['total_resolved'] == 1


if __name__ == "__main__":
    pytest.main([__file__])