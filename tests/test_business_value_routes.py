"""
Tests for Business Value Tracking API Routes

This module contains comprehensive tests for the business value tracking API endpoints
including ROI calculations, cost savings analysis, productivity measurement,
and competitive advantage assessment routes.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from scrollintel.api.routes.business_value_routes import router
from scrollintel.models.business_value_models import (
    BusinessValueMetric, ROICalculation, CostSavingsRecord,
    ProductivityRecord, CompetitiveAdvantageAssessment,
    MetricType, BusinessUnit, CompetitiveAdvantageType
)

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestBusinessValueRoutes:
    """Test suite for Business Value API routes"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return Mock()
    
    @pytest.fixture
    def sample_metric_data(self):
        """Sample business value metric data"""
        return {
            "metric_type": "roi",
            "business_unit": "sales",
            "metric_name": "Sales Process Automation ROI",
            "baseline_value": 100000,
            "current_value": 150000,
            "measurement_period_start": "2024-01-01T00:00:00",
            "measurement_period_end": "2024-03-31T23:59:59",
            "currency": "USD"
        }
    
    @pytest.fixture
    def sample_roi_data(self):
        """Sample ROI calculation data"""
        return {
            "metric_id": 1,
            "investment_amount": 100000,
            "return_amount": 150000,
            "calculation_method": "Standard ROI"
        }
    
    @pytest.fixture
    def sample_cost_savings_data(self):
        """Sample cost savings data"""
        return {
            "metric_id": 1,
            "savings_category": "Process Automation",
            "cost_before": 50000,
            "cost_after": 30000,
            "savings_source": "AI Agent Implementation"
        }
    
    @pytest.fixture
    def sample_productivity_data(self):
        """Sample productivity data"""
        return {
            "metric_id": 1,
            "task_category": "Data Analysis",
            "baseline_time_hours": 10,
            "current_time_hours": 6,
            "tasks_completed_baseline": 100,
            "tasks_completed_current": 120,
            "quality_score_baseline": 7.5,
            "quality_score_current": 8.5
        }
    
    @pytest.fixture
    def sample_competitive_advantage_data(self):
        """Sample competitive advantage data"""
        return {
            "advantage_type": "innovation",
            "competitor_name": "Competitor A",
            "our_score": 9.0,
            "competitor_score": 6.5,
            "market_impact": "HIGH",
            "sustainability_months": 18,
            "assessor": "Strategic Team"
        }
    
    def test_create_business_value_metric_success(self, sample_metric_data, mock_db_session):
        """Test successful business value metric creation"""
        mock_metric = Mock(spec=BusinessValueMetric)
        mock_metric.id = 1
        mock_metric.metric_type = "roi"
        mock_metric.business_unit = "sales"
        mock_metric.metric_name = sample_metric_data["metric_name"]
        mock_metric.baseline_value = Decimal('100000')
        mock_metric.current_value = Decimal('150000')
        mock_metric.target_value = None
        mock_metric.measurement_period_start = datetime(2024, 1, 1)
        mock_metric.measurement_period_end = datetime(2024, 3, 31, 23, 59, 59)
        mock_metric.currency = "USD"
        mock_metric.created_at = datetime.utcnow()
        mock_metric.updated_at = datetime.utcnow()
        mock_metric.metadata = {}
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.create_business_value_metric', 
                   return_value=mock_metric):
            
            response = client.post("/api/v1/business-value/metrics", json=sample_metric_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["metric_name"] == sample_metric_data["metric_name"]
            assert data["metric_type"] == "roi"
            assert data["business_unit"] == "sales"
    
    def test_create_business_value_metric_invalid_data(self):
        """Test business value metric creation with invalid data"""
        invalid_data = {
            "metric_type": "invalid_type",
            "business_unit": "sales",
            "metric_name": "",  # Empty name
            "baseline_value": -100,  # Negative value
            "current_value": 150000
        }
        
        response = client.post("/api/v1/business-value/metrics", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_get_business_value_metrics_success(self, mock_db_session):
        """Test successful retrieval of business value metrics"""
        mock_metrics = [
            Mock(
                id=1,
                metric_type="roi",
                business_unit="sales",
                metric_name="Sales ROI",
                baseline_value=Decimal('100000'),
                current_value=Decimal('150000'),
                target_value=None,
                measurement_period_start=datetime(2024, 1, 1),
                measurement_period_end=datetime(2024, 3, 31),
                currency="USD",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={}
            )
        ]
        
        mock_query = Mock()
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_metrics
        mock_db_session.query.return_value = mock_query
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session):
            response = client.get("/api/v1/business-value/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == 1
            assert data[0]["metric_name"] == "Sales ROI"
    
    def test_get_business_value_metrics_with_filters(self, mock_db_session):
        """Test retrieval of business value metrics with filters"""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        mock_db_session.query.return_value = mock_query
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session):
            response = client.get(
                "/api/v1/business-value/metrics",
                params={
                    "metric_type": "roi",
                    "business_unit": "sales",
                    "limit": 50,
                    "offset": 0
                }
            )
            
            assert response.status_code == 200
            # Verify filters were applied
            assert mock_query.filter.call_count >= 2
    
    def test_get_business_value_metric_by_id_success(self, mock_db_session):
        """Test successful retrieval of specific business value metric"""
        mock_metric = Mock(
            id=1,
            metric_type="roi",
            business_unit="sales",
            metric_name="Sales ROI",
            baseline_value=Decimal('100000'),
            current_value=Decimal('150000'),
            target_value=None,
            measurement_period_start=datetime(2024, 1, 1),
            measurement_period_end=datetime(2024, 3, 31),
            currency="USD",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={}
        )
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_metric
        mock_db_session.query.return_value = mock_query
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session):
            response = client.get("/api/v1/business-value/metrics/1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["metric_name"] == "Sales ROI"
    
    def test_get_business_value_metric_by_id_not_found(self, mock_db_session):
        """Test retrieval of non-existent business value metric"""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session):
            response = client.get("/api/v1/business-value/metrics/999")
            
            assert response.status_code == 404
    
    def test_create_roi_calculation_success(self, sample_roi_data, mock_db_session):
        """Test successful ROI calculation creation"""
        mock_roi = Mock(spec=ROICalculation)
        mock_roi.id = 1
        mock_roi.metric_id = 1
        mock_roi.investment_amount = Decimal('100000')
        mock_roi.return_amount = Decimal('150000')
        mock_roi.roi_percentage = Decimal('50.00')
        mock_roi.payback_period_months = 8
        mock_roi.npv = Decimal('45000')
        mock_roi.irr = Decimal('75.00')
        mock_roi.calculation_date = datetime.utcnow()
        mock_roi.calculation_method = "Standard ROI"
        mock_roi.confidence_level = None
        mock_roi.assumptions = {}
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.create_roi_calculation',
                   return_value=mock_roi):
            
            response = client.post("/api/v1/business-value/roi-calculations", json=sample_roi_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["roi_percentage"] == 50.00
            assert data["payback_period_months"] == 8
    
    def test_create_cost_savings_record_success(self, sample_cost_savings_data, mock_db_session):
        """Test successful cost savings record creation"""
        mock_savings = Mock(spec=CostSavingsRecord)
        mock_savings.id = 1
        mock_savings.metric_id = 1
        mock_savings.savings_category = "Process Automation"
        mock_savings.annual_savings = Decimal('20000')
        mock_savings.monthly_savings = Decimal('1666.67')
        mock_savings.cost_before = Decimal('50000')
        mock_savings.cost_after = Decimal('30000')
        mock_savings.savings_source = "AI Agent Implementation"
        mock_savings.verification_method = None
        mock_savings.verified = False
        mock_savings.verification_date = None
        mock_savings.record_date = datetime.utcnow()
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.create_cost_savings_record',
                   return_value=mock_savings):
            
            response = client.post("/api/v1/business-value/cost-savings", json=sample_cost_savings_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["savings_category"] == "Process Automation"
            assert data["annual_savings"] == 20000
    
    def test_create_productivity_record_success(self, sample_productivity_data, mock_db_session):
        """Test successful productivity record creation"""
        mock_productivity = Mock(spec=ProductivityRecord)
        mock_productivity.id = 1
        mock_productivity.metric_id = 1
        mock_productivity.task_category = "Data Analysis"
        mock_productivity.baseline_time_hours = Decimal('10')
        mock_productivity.current_time_hours = Decimal('6')
        mock_productivity.efficiency_gain_percentage = Decimal('40.00')
        mock_productivity.tasks_completed_baseline = 100
        mock_productivity.tasks_completed_current = 120
        mock_productivity.quality_score_baseline = Decimal('7.5')
        mock_productivity.quality_score_current = Decimal('8.5')
        mock_productivity.measurement_date = datetime.utcnow()
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.create_productivity_record',
                   return_value=mock_productivity):
            
            response = client.post("/api/v1/business-value/productivity", json=sample_productivity_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["task_category"] == "Data Analysis"
            assert data["efficiency_gain_percentage"] == 40.00
    
    def test_create_competitive_advantage_assessment_success(self, sample_competitive_advantage_data, mock_db_session):
        """Test successful competitive advantage assessment creation"""
        mock_assessment = Mock(spec=CompetitiveAdvantageAssessment)
        mock_assessment.id = 1
        mock_assessment.advantage_type = "innovation"
        mock_assessment.competitor_name = "Competitor A"
        mock_assessment.our_score = Decimal('9.0')
        mock_assessment.competitor_score = Decimal('6.5')
        mock_assessment.advantage_gap = Decimal('2.5')
        mock_assessment.market_impact = "HIGH"
        mock_assessment.sustainability_months = 18
        mock_assessment.assessment_date = datetime.utcnow()
        mock_assessment.assessor = "Strategic Team"
        mock_assessment.evidence = {}
        mock_assessment.action_items = []
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.create_competitive_advantage_assessment',
                   return_value=mock_assessment):
            
            response = client.post("/api/v1/business-value/competitive-advantage", json=sample_competitive_advantage_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["advantage_type"] == "innovation"
            assert data["advantage_gap"] == 2.5
    
    def test_get_business_value_summary_success(self, mock_db_session):
        """Test successful business value summary generation"""
        mock_summary = Mock()
        mock_summary.total_roi_percentage = Decimal('45.5')
        mock_summary.total_cost_savings = Decimal('150000')
        mock_summary.total_productivity_gains = Decimal('32.5')
        mock_summary.competitive_advantages_count = 5
        mock_summary.top_performing_metrics = []
        mock_summary.improvement_trends = {"roi": [Decimal('45.5')]}
        mock_summary.business_unit_performance = {"sales": Decimal('50')}
        mock_summary.report_period_start = datetime(2024, 1, 1)
        mock_summary.report_period_end = datetime(2024, 3, 31)
        mock_summary.generated_at = datetime.utcnow()
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.generate_business_value_summary',
                   return_value=mock_summary):
            
            response = client.get("/api/v1/business-value/summary")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_roi_percentage"] == 45.5
            assert data["total_cost_savings"] == 150000
            assert data["competitive_advantages_count"] == 5
    
    def test_get_business_value_dashboard_success(self, mock_db_session):
        """Test successful business value dashboard generation"""
        mock_dashboard = Mock()
        mock_dashboard.key_metrics = {
            "total_roi": Decimal('45'),
            "total_savings": Decimal('150000'),
            "productivity_gain": Decimal('30'),
            "competitive_advantages": Decimal('5')
        }
        mock_dashboard.roi_trend = [
            {"date": "2024-01-01T00:00:00", "value": 36.0},
            {"date": "2024-03-31T23:59:59", "value": 45.0}
        ]
        mock_dashboard.cost_savings_breakdown = {
            "automation": Decimal('60000'),
            "efficiency": Decimal('45000'),
            "optimization": Decimal('45000')
        }
        mock_dashboard.productivity_improvements = {"sales": Decimal('25')}
        mock_dashboard.competitive_position = {
            "overall_score": Decimal('8.5'),
            "market_position": "LEADING",
            "trend": "IMPROVING"
        }
        mock_dashboard.alerts = []
        mock_dashboard.recommendations = ["Continue monitoring high-performing metrics"]
        mock_dashboard.last_updated = datetime.utcnow()
        
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.get_business_value_dashboard',
                   return_value=mock_dashboard):
            
            response = client.get("/api/v1/business-value/dashboard")
            
            assert response.status_code == 200
            data = response.json()
            assert "key_metrics" in data
            assert "roi_trend" in data
            assert "cost_savings_breakdown" in data
    
    def test_calculate_roi_metrics_success(self):
        """Test ROI metrics calculation endpoint"""
        with patch('scrollintel.api.routes.business_value_routes.business_value_engine.calculate_roi') as mock_calc:
            mock_calc.return_value = {
                'roi_percentage': Decimal('50.00'),
                'npv': Decimal('45000.00'),
                'irr': Decimal('75.00'),
                'payback_period_months': 8
            }
            
            response = client.post(
                "/api/v1/business-value/calculate-roi",
                params={
                    "investment": 100000,
                    "returns": 150000,
                    "time_period_months": 12,
                    "discount_rate": 10
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["roi_percentage"] == 50.0
            assert data["npv"] == 45000.0
            assert data["payback_period_months"] == 8
    
    def test_calculate_cost_savings_metrics_success(self):
        """Test cost savings metrics calculation endpoint"""
        with patch('scrollintel.api.routes.business_value_routes.business_value_engine.track_cost_savings') as mock_calc:
            mock_calc.return_value = {
                'total_savings': Decimal('20000.00'),
                'savings_percentage': Decimal('40.00'),
                'annual_savings': Decimal('20000.00'),
                'monthly_savings': Decimal('1666.67')
            }
            
            response = client.post(
                "/api/v1/business-value/calculate-cost-savings",
                params={
                    "cost_before": 50000,
                    "cost_after": 30000,
                    "time_period_months": 12
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_savings"] == 20000.0
            assert data["savings_percentage"] == 40.0
            assert data["annual_savings"] == 20000.0
    
    def test_calculate_productivity_gains_metrics_success(self):
        """Test productivity gains metrics calculation endpoint"""
        with patch('scrollintel.api.routes.business_value_routes.business_value_engine.measure_productivity_gains') as mock_calc:
            mock_calc.return_value = {
                'efficiency_gain_percentage': Decimal('40.00'),
                'quality_improvement_percentage': Decimal('13.33'),
                'volume_improvement_percentage': Decimal('20.00'),
                'overall_productivity_score': Decimal('31.33'),
                'time_savings_hours': Decimal('4.00')
            }
            
            response = client.post(
                "/api/v1/business-value/calculate-productivity-gains",
                params={
                    "baseline_time": 10,
                    "current_time": 6,
                    "baseline_quality": 7.5,
                    "current_quality": 8.5,
                    "baseline_volume": 100,
                    "current_volume": 120
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["efficiency_gain_percentage"] == 40.0
            assert data["quality_improvement_percentage"] == 13.33
            assert data["volume_improvement_percentage"] == 20.0
    
    def test_api_error_handling(self, mock_db_session):
        """Test API error handling"""
        with patch('scrollintel.api.routes.business_value_routes.get_database_session', return_value=mock_db_session), \
             patch('scrollintel.api.routes.business_value_routes.business_value_engine.create_business_value_metric',
                   side_effect=Exception("Database error")):
            
            response = client.post("/api/v1/business-value/metrics", json={
                "metric_type": "roi",
                "business_unit": "sales",
                "metric_name": "Test Metric",
                "baseline_value": 100000,
                "current_value": 150000,
                "measurement_period_start": "2024-01-01T00:00:00",
                "measurement_period_end": "2024-03-31T23:59:59"
            })
            
            assert response.status_code == 400
            assert "Database error" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main([__file__])