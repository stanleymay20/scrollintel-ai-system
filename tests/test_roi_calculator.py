"""
Unit tests for ROI Calculator Engine.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import uuid

from scrollintel.engines.roi_calculator import (
    ROICalculator, ROICalculationResult, CostSummary, BenefitSummary
)
from scrollintel.models.roi_models import (
    ROIAnalysis, CostTracking, BenefitTracking, ROIReport,
    CloudCostCollection, ProductivityMetric,
    CostType, BenefitType, ProjectStatus
)


class TestROICalculator:
    """Test suite for ROI Calculator functionality."""
    
    @pytest.fixture
    def roi_calculator(self):
        """Create ROI calculator instance for testing."""
        return ROICalculator()
    
    @pytest.fixture
    def sample_project_data(self):
        """Sample project data for testing."""
        return {
            'project_id': 'test-project-001',
            'project_name': 'Test AI Implementation',
            'project_description': 'Test project for ROI calculation',
            'project_start_date': datetime.utcnow() - timedelta(days=90),
            'analysis_period_months': 12
        }
    
    @pytest.fixture
    def sample_costs(self):
        """Sample cost data for testing."""
        return [
            {
                'cost_category': 'infrastructure',
                'description': 'Cloud computing costs',
                'amount': 5000.0,
                'vendor': 'AWS',
                'is_recurring': True,
                'recurrence_frequency': 'monthly'
            },
            {
                'cost_category': 'personnel',
                'description': 'Development team costs',
                'amount': 15000.0,
                'vendor': 'Internal',
                'is_recurring': False
            }
        ]
    
    @pytest.fixture
    def sample_benefits(self):
        """Sample benefit data for testing."""
        return [
            {
                'benefit_category': 'productivity_gain',
                'description': 'Automated data processing',
                'quantified_value': 8000.0,
                'measurement_method': 'time_study',
                'baseline_value': 40.0,
                'current_value': 10.0,
                'is_realized': True
            },
            {
                'benefit_category': 'cost_savings',
                'description': 'Reduced manual processing',
                'quantified_value': 12000.0,
                'measurement_method': 'cost_comparison',
                'is_realized': True
            }
        ]
    
    def test_create_roi_analysis(self, roi_calculator, sample_project_data):
        """Test creating a new ROI analysis."""
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = None
            
            # Mock ROIAnalysis creation
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_roi_analysis.project_id = sample_project_data['project_id']
            mock_roi_analysis.project_name = sample_project_data['project_name']
            
            result = roi_calculator.create_roi_analysis(**sample_project_data)
            
            # Verify session operations
            mock_session_instance.add.assert_called_once()
            mock_session_instance.commit.assert_called_once()
    
    def test_track_project_costs(self, roi_calculator, sample_costs):
        """Test tracking project costs."""
        project_id = 'test-project-001'
        
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock ROI analysis exists
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
            
            # Mock cost tracking
            mock_cost_item = MagicMock()
            mock_cost_item.id = str(uuid.uuid4())
            mock_cost_item.amount = sample_costs[0]['amount']
            
            with patch.object(roi_calculator, '_update_roi_totals'):
                result = roi_calculator.track_project_costs(
                    project_id=project_id,
                    **sample_costs[0]
                )
                
                mock_session_instance.add.assert_called_once()
                mock_session_instance.commit.assert_called_once()
    
    def test_track_project_benefits(self, roi_calculator, sample_benefits):
        """Test tracking project benefits."""
        project_id = 'test-project-001'
        
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock ROI analysis exists
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
            
            # Mock benefit tracking
            mock_benefit_item = MagicMock()
            mock_benefit_item.id = str(uuid.uuid4())
            mock_benefit_item.quantified_value = sample_benefits[0]['quantified_value']
            
            with patch.object(roi_calculator, '_update_roi_totals'):
                result = roi_calculator.track_project_benefits(
                    project_id=project_id,
                    **sample_benefits[0]
                )
                
                mock_session_instance.add.assert_called_once()
                mock_session_instance.commit.assert_called_once()
    
    def test_calculate_roi(self, roi_calculator):
        """Test ROI calculation."""
        project_id = 'test-project-001'
        
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock ROI analysis
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_roi_analysis.project_start_date = datetime.utcnow() - timedelta(days=90)
            mock_roi_analysis.confidence_level = 0.8
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
            
            # Mock costs and benefits
            mock_costs = [MagicMock(amount=10000.0), MagicMock(amount=5000.0)]
            mock_benefits = [
                MagicMock(quantified_value=12000.0, is_realized=True),
                MagicMock(quantified_value=8000.0, is_realized=True)
            ]
            
            mock_session_instance.query.return_value.filter_by.return_value.all.side_effect = [mock_costs, mock_benefits]
            
            result = roi_calculator.calculate_roi(project_id)
            
            assert isinstance(result, ROICalculationResult)
            assert result.total_investment == 15000.0
            assert result.total_benefits == 20000.0
            assert result.roi_percentage > 0
    
    def test_measure_efficiency_gains(self, roi_calculator):
        """Test measuring efficiency gains."""
        project_id = 'test-project-001'
        
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock efficiency metric creation
            mock_efficiency_metric = MagicMock()
            mock_efficiency_metric.id = str(uuid.uuid4())
            mock_efficiency_metric.time_saved_hours = 30.0
            mock_efficiency_metric.time_saved_percentage = 75.0
            mock_efficiency_metric.monthly_savings = 1500.0
            mock_efficiency_metric.annual_savings = 18000.0
            
            with patch.object(roi_calculator, 'track_project_benefits'):
                result = roi_calculator.measure_efficiency_gains(
                    project_id=project_id,
                    process_name='Data Processing',
                    time_before_hours=40.0,
                    time_after_hours=10.0,
                    frequency_per_month=1.0,
                    hourly_rate=50.0
                )
                
                mock_session_instance.add.assert_called_once()
                mock_session_instance.commit.assert_called_once()
    
    def test_generate_detailed_roi_breakdown(self, roi_calculator):
        """Test generating detailed ROI breakdown."""
        project_id = 'test-project-001'
        
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock ROI analysis
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
            
            # Mock costs and benefits with dates
            mock_costs = [
                MagicMock(amount=5000.0, cost_date=datetime(2024, 1, 15)),
                MagicMock(amount=3000.0, cost_date=datetime(2024, 2, 15))
            ]
            mock_benefits = [
                MagicMock(quantified_value=8000.0, benefit_date=datetime(2024, 2, 15), is_realized=True),
                MagicMock(quantified_value=4000.0, benefit_date=datetime(2024, 3, 15), is_realized=True)
            ]
            
            mock_session_instance.query.return_value.filter_by.return_value.all.side_effect = [mock_costs, mock_benefits]
            
            result = roi_calculator.generate_detailed_roi_breakdown(project_id)
            
            assert 'project_id' in result
            assert 'monthly_trends' in result
            assert 'cumulative_roi_trend' in result
            assert 'risk_analysis' in result
            assert 'sensitivity_analysis' in result
    
    def test_get_cost_summary(self, roi_calculator):
        """Test getting cost summary."""
        project_id = 'test-project-001'
        
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock ROI analysis
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
            
            # Mock costs
            mock_costs = [
                MagicMock(amount=5000.0, cost_category='infrastructure', is_recurring=True, recurrence_frequency='monthly'),
                MagicMock(amount=10000.0, cost_category='personnel', is_recurring=False, recurrence_frequency=None)
            ]
            mock_session_instance.query.return_value.filter_by.return_value.all.return_value = mock_costs
            
            result = roi_calculator.get_cost_summary(project_id)
            
            assert isinstance(result, CostSummary)
            assert result.total_costs == 15000.0
            assert result.monthly_recurring_costs == 5000.0
    
    def test_get_benefit_summary(self, roi_calculator):
        """Test getting benefit summary."""
        project_id = 'test-project-001'
        
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock ROI analysis
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
            
            # Mock benefits
            mock_benefits = [
                MagicMock(quantified_value=8000.0, benefit_category='productivity_gain', is_realized=True),
                MagicMock(quantified_value=4000.0, benefit_category='cost_savings', is_realized=False)
            ]
            mock_session_instance.query.return_value.filter_by.return_value.all.return_value = mock_benefits
            
            result = roi_calculator.get_benefit_summary(project_id)
            
            assert isinstance(result, BenefitSummary)
            assert result.total_benefits == 12000.0
            assert result.realized_benefits == 8000.0
            assert result.projected_benefits == 4000.0
    
    def test_generate_roi_report(self, roi_calculator):
        """Test generating ROI report."""
        project_id = 'test-project-001'
        
        with patch.object(roi_calculator, 'calculate_roi') as mock_calculate:
            with patch.object(roi_calculator, 'get_cost_summary') as mock_cost_summary:
                with patch.object(roi_calculator, 'get_benefit_summary') as mock_benefit_summary:
                    with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
                        # Mock return values
                        mock_calculate.return_value = ROICalculationResult(
                            roi_percentage=33.3,
                            net_present_value=5000.0,
                            internal_rate_of_return=0.25,
                            payback_period_months=18,
                            break_even_date=datetime.utcnow() + timedelta(days=540),
                            total_investment=15000.0,
                            total_benefits=20000.0,
                            confidence_level=0.8,
                            calculation_date=datetime.utcnow()
                        )
                        
                        mock_cost_summary.return_value = CostSummary(
                            total_costs=15000.0,
                            direct_costs=10000.0,
                            indirect_costs=5000.0,
                            operational_costs=0.0,
                            infrastructure_costs=5000.0,
                            personnel_costs=10000.0,
                            cost_breakdown={'infrastructure': 5000.0, 'personnel': 10000.0},
                            monthly_recurring_costs=1000.0
                        )
                        
                        mock_benefit_summary.return_value = BenefitSummary(
                            total_benefits=20000.0,
                            realized_benefits=15000.0,
                            projected_benefits=5000.0,
                            cost_savings=8000.0,
                            productivity_gains=12000.0,
                            revenue_increases=0.0,
                            benefit_breakdown={'productivity_gain': 12000.0, 'cost_savings': 8000.0},
                            realization_percentage=75.0
                        )
                        
                        # Mock database session
                        mock_session_instance = MagicMock()
                        mock_session.return_value.__enter__.return_value = mock_session_instance
                        
                        mock_roi_analysis = MagicMock()
                        mock_roi_analysis.id = str(uuid.uuid4())
                        mock_roi_analysis.project_name = 'Test Project'
                        mock_roi_analysis.analysis_period_start = datetime.utcnow() - timedelta(days=90)
                        mock_roi_analysis.analysis_period_end = datetime.utcnow() + timedelta(days=275)
                        mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
                        
                        mock_report = MagicMock()
                        mock_report.id = str(uuid.uuid4())
                        mock_report.report_name = 'ROI Report - Test Project'
                        
                        result = roi_calculator.generate_roi_report(project_id)
                        
                        mock_session_instance.add.assert_called_once()
                        mock_session_instance.commit.assert_called_once()
    
    def test_cloud_cost_collection(self, roi_calculator):
        """Test automated cloud cost collection."""
        project_id = 'test-project-001'
        
        with patch.object(roi_calculator, 'cloud_connector') as mock_connector:
            with patch.object(roi_calculator, 'track_project_costs') as mock_track_costs:
                with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
                    # Mock cloud costs data
                    mock_cloud_costs = [
                        {
                            'service_name': 'EC2',
                            'cost': 150.0,
                            'billing_period': '2024-01',
                            'usage_start_date': datetime(2024, 1, 1),
                            'usage_end_date': datetime(2024, 1, 31)
                        },
                        {
                            'service_name': 'S3',
                            'cost': 25.0,
                            'billing_period': '2024-01',
                            'usage_start_date': datetime(2024, 1, 1),
                            'usage_end_date': datetime(2024, 1, 31)
                        }
                    ]
                    
                    mock_connector.get_costs.return_value = mock_cloud_costs
                    
                    mock_session_instance = MagicMock()
                    mock_session.return_value.__enter__.return_value = mock_session_instance
                    
                    result = roi_calculator.collect_cloud_costs(
                        project_id=project_id,
                        provider='aws',
                        account_id='123456789',
                        start_date=datetime(2024, 1, 1),
                        end_date=datetime(2024, 1, 31)
                    )
                    
                    assert len(result) == 2
                    assert mock_track_costs.call_count == 2
                    mock_session_instance.commit.assert_called_once()
    
    def test_roi_calculation_accuracy(self, roi_calculator):
        """Test ROI calculation accuracy with known values."""
        # Test with specific values to verify calculation accuracy
        costs = [MagicMock(amount=10000.0)]  # $10,000 investment
        benefits = [MagicMock(quantified_value=15000.0, is_realized=True)]  # $15,000 benefit
        
        # Expected ROI: (15000 - 10000) / 10000 * 100 = 50%
        with patch('scrollintel.engines.roi_calculator.get_sync_db') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            mock_roi_analysis = MagicMock()
            mock_roi_analysis.id = str(uuid.uuid4())
            mock_roi_analysis.project_start_date = datetime.utcnow() - timedelta(days=90)
            mock_roi_analysis.confidence_level = 0.8
            mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_roi_analysis
            mock_session_instance.query.return_value.filter_by.return_value.all.side_effect = [costs, benefits]
            
            result = roi_calculator.calculate_roi('test-project')
            
            assert result.total_investment == 10000.0
            assert result.total_benefits == 15000.0
            assert result.roi_percentage == 50.0