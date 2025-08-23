"""
Tests for Business Value Tracking Engine

This module contains comprehensive tests for the business value tracking engine
including ROI calculations, cost savings analysis, productivity measurement,
and competitive advantage assessment.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.orm import Session

from scrollintel.engines.business_value_engine import BusinessValueEngine
from scrollintel.models.business_value_models import (
    BusinessValueMetric, ROICalculation, CostSavingsRecord,
    ProductivityRecord, CompetitiveAdvantageAssessment,
    BusinessValueMetricCreate, ROICalculationCreate, CostSavingsCreate,
    ProductivityCreate, CompetitiveAdvantageCreate,
    MetricType, BusinessUnit, CompetitiveAdvantageType
)

class TestBusinessValueEngine:
    """Test suite for BusinessValueEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create BusinessValueEngine instance"""
        return BusinessValueEngine()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.mark.asyncio
    async def test_calculate_roi_basic(self, engine):
        """Test basic ROI calculation"""
        investment = Decimal('100000')
        returns = Decimal('150000')
        
        result = await engine.calculate_roi(investment, returns)
        
        assert result['roi_percentage'] == Decimal('50.00')
        assert result['payback_period_months'] == 8  # 100000 / (150000/12)
        assert 'npv' in result
        assert 'irr' in result
    
    @pytest.mark.asyncio
    async def test_calculate_roi_with_discount_rate(self, engine):
        """Test ROI calculation with NPV"""
        investment = Decimal('100000')
        returns = Decimal('150000')
        discount_rate = Decimal('10')  # 10% annual
        
        result = await engine.calculate_roi(investment, returns, 12, discount_rate)
        
        assert result['roi_percentage'] == Decimal('50.00')
        assert result['npv'] is not None
        assert result['npv'] < returns - investment  # NPV should be less than simple profit
    
    @pytest.mark.asyncio
    async def test_calculate_roi_zero_investment_error(self, engine):
        """Test ROI calculation with zero investment raises error"""
        with pytest.raises(ValueError):
            await engine.calculate_roi(Decimal('0'), Decimal('100000'))
    
    @pytest.mark.asyncio
    async def test_track_cost_savings(self, engine):
        """Test cost savings calculation"""
        cost_before = Decimal('50000')
        cost_after = Decimal('30000')
        
        result = await engine.track_cost_savings(cost_before, cost_after)
        
        assert result['total_savings'] == Decimal('20000.00')
        assert result['savings_percentage'] == Decimal('40.00')
        assert result['annual_savings'] == Decimal('20000.00')
        assert result['monthly_savings'] == Decimal('1666.67')
    
    @pytest.mark.asyncio
    async def test_track_cost_savings_different_period(self, engine):
        """Test cost savings calculation with different time period"""
        cost_before = Decimal('50000')
        cost_after = Decimal('30000')
        time_period = 6  # 6 months
        
        result = await engine.track_cost_savings(cost_before, cost_after, time_period)
        
        assert result['total_savings'] == Decimal('20000.00')
        assert result['annual_savings'] == Decimal('40000.00')  # Annualized
        assert result['monthly_savings'] == Decimal('3333.33')
    
    @pytest.mark.asyncio
    async def test_measure_productivity_gains_time_only(self, engine):
        """Test productivity measurement with time metrics only"""
        baseline_time = Decimal('10')
        current_time = Decimal('6')
        
        result = await engine.measure_productivity_gains(baseline_time, current_time)
        
        assert result['efficiency_gain_percentage'] == Decimal('40.00')
        assert result['time_savings_hours'] == Decimal('4.00')
        assert result['quality_improvement_percentage'] == Decimal('0.00')
        assert result['volume_improvement_percentage'] == Decimal('0.00')
    
    @pytest.mark.asyncio
    async def test_measure_productivity_gains_comprehensive(self, engine):
        """Test comprehensive productivity measurement"""
        baseline_time = Decimal('10')
        current_time = Decimal('6')
        baseline_quality = Decimal('7')
        current_quality = Decimal('8.5')
        baseline_volume = 100
        current_volume = 120
        
        result = await engine.measure_productivity_gains(
            baseline_time, current_time, baseline_quality, current_quality,
            baseline_volume, current_volume
        )
        
        assert result['efficiency_gain_percentage'] == Decimal('40.00')
        assert result['quality_improvement_percentage'] == Decimal('21.43')
        assert result['volume_improvement_percentage'] == Decimal('20.00')
        assert result['overall_productivity_score'] > Decimal('30.00')  # Weighted average
    
    @pytest.mark.asyncio
    async def test_assess_competitive_advantage(self, engine):
        """Test competitive advantage assessment"""
        our_capabilities = {
            'innovation': Decimal('9'),
            'speed': Decimal('8'),
            'cost': Decimal('7')
        }
        competitor_capabilities = {
            'innovation': Decimal('6'),
            'speed': Decimal('7'),
            'cost': Decimal('8')
        }
        market_weights = {
            'innovation': Decimal('3'),
            'speed': Decimal('2'),
            'cost': Decimal('1')
        }
        
        result = await engine.assess_competitive_advantage(
            our_capabilities, competitor_capabilities, market_weights
        )
        
        assert 'capability_advantages' in result
        assert result['overall_our_score'] > result['overall_competitor_score']
        assert result['market_impact'] in ['HIGH', 'MEDIUM', 'LOW']
        assert result['sustainability_months'] >= 6
        assert result['competitive_strength'] in ['DOMINANT', 'STRONG', 'MODERATE', 'WEAK', 'DISADVANTAGED']
    
    @pytest.mark.asyncio
    async def test_create_business_value_metric(self, engine, mock_db):
        """Test creating business value metric"""
        metric_data = BusinessValueMetricCreate(
            metric_type=MetricType.ROI,
            business_unit=BusinessUnit.SALES,
            metric_name="Sales Process Automation ROI",
            baseline_value=Decimal('100000'),
            current_value=Decimal('150000'),
            measurement_period_start=datetime.utcnow() - timedelta(days=30),
            measurement_period_end=datetime.utcnow()
        )
        
        # Mock database operations
        mock_metric = Mock(spec=BusinessValueMetric)
        mock_metric.id = 1
        mock_metric.metric_name = metric_data.metric_name
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        with patch.object(BusinessValueMetric, '__new__', return_value=mock_metric):
            result = await engine.create_business_value_metric(metric_data, mock_db)
        
        assert result == mock_metric
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_roi_calculation(self, engine, mock_db):
        """Test creating ROI calculation"""
        roi_data = ROICalculationCreate(
            metric_id=1,
            investment_amount=Decimal('100000'),
            return_amount=Decimal('150000'),
            calculation_method="Standard ROI"
        )
        
        # Mock database operations
        mock_roi = Mock(spec=ROICalculation)
        mock_roi.id = 1
        mock_roi.roi_percentage = Decimal('50.00')
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        with patch.object(ROICalculation, '__new__', return_value=mock_roi):
            result = await engine.create_roi_calculation(roi_data, mock_db)
        
        assert result == mock_roi
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_cost_savings_record(self, engine, mock_db):
        """Test creating cost savings record"""
        savings_data = CostSavingsCreate(
            metric_id=1,
            savings_category="Process Automation",
            cost_before=Decimal('50000'),
            cost_after=Decimal('30000')
        )
        
        # Mock database operations
        mock_savings = Mock(spec=CostSavingsRecord)
        mock_savings.id = 1
        mock_savings.annual_savings = Decimal('20000')
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        with patch.object(CostSavingsRecord, '__new__', return_value=mock_savings):
            result = await engine.create_cost_savings_record(savings_data, mock_db)
        
        assert result == mock_savings
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_productivity_record(self, engine, mock_db):
        """Test creating productivity record"""
        productivity_data = ProductivityCreate(
            metric_id=1,
            task_category="Data Analysis",
            baseline_time_hours=Decimal('10'),
            current_time_hours=Decimal('6')
        )
        
        # Mock database operations
        mock_productivity = Mock(spec=ProductivityRecord)
        mock_productivity.id = 1
        mock_productivity.efficiency_gain_percentage = Decimal('40.00')
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        with patch.object(ProductivityRecord, '__new__', return_value=mock_productivity):
            result = await engine.create_productivity_record(productivity_data, mock_db)
        
        assert result == mock_productivity
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_competitive_advantage_assessment(self, engine, mock_db):
        """Test creating competitive advantage assessment"""
        advantage_data = CompetitiveAdvantageCreate(
            advantage_type=CompetitiveAdvantageType.INNOVATION,
            competitor_name="Competitor A",
            our_score=Decimal('9'),
            competitor_score=Decimal('6'),
            market_impact="HIGH"
        )
        
        # Mock database operations
        mock_assessment = Mock(spec=CompetitiveAdvantageAssessment)
        mock_assessment.id = 1
        mock_assessment.advantage_gap = Decimal('3')
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        
        with patch.object(CompetitiveAdvantageAssessment, '__new__', return_value=mock_assessment):
            result = await engine.create_competitive_advantage_assessment(advantage_data, mock_db)
        
        assert result == mock_assessment
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_business_value_summary(self, engine, mock_db):
        """Test generating business value summary"""
        start_date = datetime.utcnow() - timedelta(days=90)
        end_date = datetime.utcnow()
        
        # Mock database queries
        mock_metrics = [
            Mock(
                id=1,
                metric_type='roi',
                business_unit='sales',
                baseline_value=Decimal('100'),
                current_value=Decimal('150')
            )
        ]
        mock_roi_calcs = [Mock(roi_percentage=Decimal('50'))]
        mock_cost_savings = [Mock(annual_savings=Decimal('20000'))]
        mock_productivity = [Mock(efficiency_gain_percentage=Decimal('25'))]
        
        mock_db.query.return_value.filter.return_value.all.return_value = mock_metrics
        mock_db.query.return_value.join.return_value.filter.return_value.all.side_effect = [
            mock_roi_calcs, mock_cost_savings, mock_productivity
        ]
        mock_db.query.return_value.filter.return_value.count.return_value = 3
        
        with patch('scrollintel.engines.business_value_engine.get_database_session', return_value=mock_db):
            result = await engine.generate_business_value_summary(start_date, end_date, None, mock_db)
        
        assert result.total_roi_percentage == Decimal('50')
        assert result.total_cost_savings == Decimal('20000')
        assert result.total_productivity_gains == Decimal('25')
        assert result.competitive_advantages_count == 3
    
    @pytest.mark.asyncio
    async def test_get_business_value_dashboard(self, engine, mock_db):
        """Test getting business value dashboard"""
        # Mock the summary generation
        mock_summary = Mock()
        mock_summary.total_roi_percentage = Decimal('45')
        mock_summary.total_cost_savings = Decimal('150000')
        mock_summary.total_productivity_gains = Decimal('30')
        mock_summary.competitive_advantages_count = 5
        mock_summary.business_unit_performance = {'sales': Decimal('25'), 'marketing': Decimal('35')}
        
        with patch.object(engine, 'generate_business_value_summary', return_value=mock_summary):
            result = await engine.get_business_value_dashboard(None, mock_db)
        
        assert 'key_metrics' in result.__dict__
        assert 'roi_trend' in result.__dict__
        assert 'cost_savings_breakdown' in result.__dict__
        assert 'productivity_improvements' in result.__dict__
        assert 'competitive_position' in result.__dict__
        assert 'alerts' in result.__dict__
        assert 'recommendations' in result.__dict__
    
    def test_calculate_competitive_strength(self, engine):
        """Test competitive strength calculation"""
        assert engine._calculate_competitive_strength(Decimal('3.5')) == "DOMINANT"
        assert engine._calculate_competitive_strength(Decimal('2.5')) == "STRONG"
        assert engine._calculate_competitive_strength(Decimal('1.5')) == "MODERATE"
        assert engine._calculate_competitive_strength(Decimal('0.5')) == "WEAK"
        assert engine._calculate_competitive_strength(Decimal('-0.5')) == "DISADVANTAGED"
    
    def test_calculate_improvement_trends(self, engine):
        """Test improvement trends calculation"""
        metrics = [
            Mock(
                metric_type='roi',
                baseline_value=Decimal('100'),
                current_value=Decimal('150')
            ),
            Mock(
                metric_type='roi',
                baseline_value=Decimal('200'),
                current_value=Decimal('220')
            ),
            Mock(
                metric_type='cost_savings',
                baseline_value=Decimal('50000'),
                current_value=Decimal('40000')
            )
        ]
        
        trends = engine._calculate_improvement_trends(metrics)
        
        assert 'roi' in trends
        assert 'cost_savings' in trends
        assert len(trends['roi']) == 2
        assert trends['roi'][0] == Decimal('50')  # (150-100)/100 * 100
        assert trends['roi'][1] == Decimal('10')  # (220-200)/200 * 100
    
    def test_calculate_business_unit_performance(self, engine):
        """Test business unit performance calculation"""
        metrics = [
            Mock(
                business_unit='sales',
                baseline_value=Decimal('100'),
                current_value=Decimal('150')
            ),
            Mock(
                business_unit='sales',
                baseline_value=Decimal('200'),
                current_value=Decimal('220')
            ),
            Mock(
                business_unit='marketing',
                baseline_value=Decimal('50'),
                current_value=Decimal('75')
            )
        ]
        
        performance = engine._calculate_business_unit_performance(metrics)
        
        assert 'sales' in performance
        assert 'marketing' in performance
        assert performance['sales'] == Decimal('30')  # Average of 50% and 10%
        assert performance['marketing'] == Decimal('50')  # (75-50)/50 * 100
    
    def test_generate_alerts(self, engine):
        """Test alert generation"""
        summary = Mock()
        summary.total_roi_percentage = Decimal('5')  # Below threshold
        summary.total_productivity_gains = Decimal('3')  # Below threshold
        
        alerts = engine._generate_alerts(summary)
        
        assert len(alerts) == 2
        assert any('ROI below target' in alert['message'] for alert in alerts)
        assert any('Productivity gains opportunity' in alert['message'] for alert in alerts)
    
    def test_generate_recommendations(self, engine):
        """Test recommendation generation"""
        summary = Mock()
        summary.total_cost_savings = Decimal('50000')  # Below threshold
        summary.competitive_advantages_count = 2  # Below threshold
        
        recommendations = engine._generate_recommendations(summary)
        
        assert len(recommendations) >= 2
        assert any('automation initiatives' in rec for rec in recommendations)
        assert any('competitive advantages' in rec for rec in recommendations)

if __name__ == "__main__":
    pytest.main([__file__])