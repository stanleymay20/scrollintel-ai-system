"""
Unit tests for Dashboard Manager functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from scrollintel.core.dashboard_manager import (
    DashboardManager, DashboardConfig, SharePermissions, TimeRange, DashboardData
)
from scrollintel.models.dashboard_models import (
    Dashboard, Widget, DashboardPermission, DashboardTemplate, BusinessMetric,
    DashboardType, ExecutiveRole, WidgetType
)


class TestDashboardManager:
    """Test cases for DashboardManager class."""
    
    @pytest.fixture
    def dashboard_manager(self):
        """Create a DashboardManager instance for testing."""
        return DashboardManager()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('scrollintel.core.dashboard_manager.get_sync_db') as mock_session:
            mock_db = Mock(spec=Session)
            mock_session.return_value.__enter__.return_value = mock_db
            mock_session.return_value.__exit__.return_value = None
            yield mock_db
    
    @pytest.fixture
    def sample_dashboard_config(self):
        """Sample dashboard configuration."""
        return DashboardConfig(
            layout={"grid_columns": 12, "grid_rows": 8},
            theme="executive",
            auto_refresh=True,
            refresh_interval=300
        )
    
    def test_dashboard_config_creation(self):
        """Test DashboardConfig creation and serialization."""
        config = DashboardConfig(
            layout={"columns": 12},
            theme="dark",
            auto_refresh=False,
            refresh_interval=600
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["layout"] == {"columns": 12}
        assert config_dict["theme"] == "dark"
        assert config_dict["auto_refresh"] is False
        assert config_dict["refresh_interval"] == 600
    
    def test_create_executive_dashboard(self, dashboard_manager, mock_db_session, sample_dashboard_config):
        """Test creating an executive dashboard."""
        # Mock database operations
        mock_dashboard = Mock(spec=Dashboard)
        mock_dashboard.id = "test_dashboard_id"
        mock_dashboard.name = "CTO Executive Dashboard"
        
        mock_db_session.add.return_value = None
        mock_db_session.flush.return_value = None
        mock_db_session.commit.return_value = None
        mock_db_session.refresh.return_value = None
        
        # Mock the _get_default_widgets_for_role method
        with patch.object(dashboard_manager, '_get_default_widgets_for_role') as mock_widgets:
            mock_widgets.return_value = [
                {
                    "dashboard_id": "test_dashboard_id",
                    "name": "Test Widget",
                    "type": WidgetType.KPI.value,
                    "position_x": 0,
                    "position_y": 0,
                    "width": 4,
                    "height": 2
                }
            ]
            
            # Create dashboard
            result = dashboard_manager.create_executive_dashboard(
                ExecutiveRole.CTO, sample_dashboard_config, "user123", "Test Dashboard"
            )
            
            # Verify database operations
            assert mock_db_session.add.call_count >= 1  # Dashboard + widgets
            mock_db_session.flush.assert_called_once()
            mock_db_session.commit.assert_called_once()
    
    def test_create_dashboard_from_template(self, dashboard_manager, mock_db_session):
        """Test creating a dashboard from a template."""
        # Mock template
        mock_template = Mock(spec=DashboardTemplate)
        mock_template.id = "template_id"
        mock_template.name = "CTO Template"
        mock_template.template_config = {
            "widgets": [
                {
                    "name": "Template Widget",
                    "type": WidgetType.CHART.value,
                    "position_x": 0,
                    "position_y": 0,
                    "width": 6,
                    "height": 3
                }
            ]
        }
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_template
        mock_db_session.add.return_value = None
        mock_db_session.flush.return_value = None
        mock_db_session.commit.return_value = None
        mock_db_session.refresh.return_value = None
        
        # Create dashboard from template
        result = dashboard_manager.create_dashboard_from_template(
            "template_id", "user123", "My Dashboard"
        )
        
        # Verify template was queried
        mock_db_session.query.assert_called()
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called_once()
    
    def test_create_dashboard_from_nonexistent_template(self, dashboard_manager, mock_db_session):
        """Test creating dashboard from non-existent template raises error."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ValueError, match="Template template_id not found"):
            dashboard_manager.create_dashboard_from_template("template_id", "user123")
    
    def test_update_dashboard_metrics(self, dashboard_manager, mock_db_session):
        """Test updating dashboard metrics."""
        # Mock dashboard
        mock_dashboard = Mock(spec=Dashboard)
        mock_dashboard.id = "dashboard_id"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_dashboard
        
        metrics_data = [
            {
                "name": "revenue",
                "category": "financial",
                "value": 100000,
                "unit": "USD",
                "source": "financial_system"
            }
        ]
        
        # Update metrics
        result = dashboard_manager.update_dashboard_metrics("dashboard_id", metrics_data)
        
        assert result is True
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called_once()
    
    def test_update_dashboard_metrics_nonexistent_dashboard(self, dashboard_manager, mock_db_session):
        """Test updating metrics for non-existent dashboard."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        result = dashboard_manager.update_dashboard_metrics("nonexistent", [])
        
        assert result is False
    
    def test_get_dashboard_data(self, dashboard_manager, mock_db_session):
        """Test retrieving dashboard data."""
        # Mock dashboard with widgets
        mock_dashboard = Mock(spec=Dashboard)
        mock_dashboard.id = "dashboard_id"
        mock_dashboard.name = "Test Dashboard"
        
        mock_widget = Mock(spec=Widget)
        mock_widget.id = "widget_id"
        mock_widget.name = "Test Widget"
        mock_widget.type = WidgetType.KPI.value
        mock_widget.position_x = 0
        mock_widget.position_y = 0
        mock_widget.width = 4
        mock_widget.height = 2
        mock_widget.config = {"test": "config"}
        mock_widget.data_source = "test_source"
        mock_widget.updated_at = datetime.utcnow()
        mock_widget.is_active = True
        
        mock_dashboard.widgets = [mock_widget]
        
        # Mock metrics query
        mock_metric = Mock(spec=BusinessMetric)
        mock_metric.id = "metric_id"
        mock_metric.name = "test_metric"
        mock_metric.value = 100
        mock_metric.timestamp = datetime.utcnow()
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_dashboard
        mock_metrics_query = Mock()
        mock_metrics_query.order_by.return_value.all.return_value = [mock_metric]
        mock_db_session.query.return_value.filter.return_value = mock_metrics_query
        
        # Get dashboard data
        result = dashboard_manager.get_dashboard_data("dashboard_id")
        
        assert result is not None
        assert isinstance(result, DashboardData)
        assert result.dashboard == mock_dashboard
        assert "widget_id" in result.widgets_data
        assert len(result.metrics) == 1
    
    def test_get_dashboard_data_with_time_range(self, dashboard_manager, mock_db_session):
        """Test retrieving dashboard data with time range filter."""
        mock_dashboard = Mock(spec=Dashboard)
        mock_dashboard.id = "dashboard_id"
        mock_dashboard.widgets = []
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_dashboard
        
        # Mock metrics query with time range
        mock_metrics_query = Mock()
        mock_metrics_query.filter.return_value = mock_metrics_query
        mock_metrics_query.order_by.return_value.all.return_value = []
        mock_db_session.query.return_value.filter.return_value = mock_metrics_query
        
        time_range = TimeRange(
            start=datetime.utcnow() - timedelta(days=7),
            end=datetime.utcnow()
        )
        
        result = dashboard_manager.get_dashboard_data("dashboard_id", time_range)
        
        assert result is not None
        # Verify time range filter was applied
        mock_metrics_query.filter.assert_called()
    
    def test_share_dashboard(self, dashboard_manager, mock_db_session):
        """Test sharing a dashboard."""
        mock_dashboard = Mock(spec=Dashboard)
        mock_dashboard.id = "dashboard_id"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_dashboard
        
        permissions = SharePermissions(
            users=["user1", "user2"],
            permission_type="view",
            expires_in_days=30
        )
        
        # Share dashboard
        result = dashboard_manager.share_dashboard("dashboard_id", permissions, "owner_id")
        
        assert result.url.endswith("/dashboard/dashboard_id/shared")
        assert result.expires_at is not None
        assert mock_db_session.add.call_count == 2  # Two users
        mock_db_session.commit.assert_called_once()
    
    def test_share_nonexistent_dashboard(self, dashboard_manager, mock_db_session):
        """Test sharing non-existent dashboard raises error."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        permissions = SharePermissions(users=["user1"], permission_type="view")
        
        with pytest.raises(ValueError, match="Dashboard dashboard_id not found"):
            dashboard_manager.share_dashboard("dashboard_id", permissions, "owner_id")
    
    def test_get_dashboards_for_user(self, dashboard_manager, mock_db_session):
        """Test retrieving dashboards for a user."""
        mock_dashboards = [Mock(spec=Dashboard) for _ in range(3)]
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value.all.return_value = mock_dashboards
        mock_db_session.query.return_value = mock_query
        
        result = dashboard_manager.get_dashboards_for_user("user123")
        
        assert len(result) == 3
        mock_db_session.query.assert_called()
    
    def test_get_dashboards_for_user_with_role_filter(self, dashboard_manager, mock_db_session):
        """Test retrieving dashboards for a user with role filter."""
        mock_dashboards = [Mock(spec=Dashboard)]
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value.all.return_value = mock_dashboards
        mock_db_session.query.return_value = mock_query
        
        result = dashboard_manager.get_dashboards_for_user("user123", "cto")
        
        assert len(result) == 1
        # Verify role filter was applied
        assert mock_query.filter.call_count >= 2  # User filter + role filter
    
    def test_create_dashboard_template(self, dashboard_manager, mock_db_session):
        """Test creating a dashboard template."""
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None
        mock_db_session.refresh.return_value = None
        
        template_config = {"widgets": [], "layout": {"columns": 12}}
        
        result = dashboard_manager.create_dashboard_template(
            "Test Template", "executive", "cto", "Test description", 
            template_config, "creator_id"
        )
        
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_get_templates_for_role(self, dashboard_manager, mock_db_session):
        """Test retrieving templates for a specific role."""
        mock_templates = [Mock(spec=DashboardTemplate) for _ in range(2)]
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value.all.return_value = mock_templates
        mock_db_session.query.return_value = mock_query
        
        result = dashboard_manager.get_templates_for_role("cto")
        
        assert len(result) == 2
        mock_db_session.query.assert_called()
    
    def test_get_default_widgets_for_cto(self, dashboard_manager):
        """Test getting default widgets for CTO role."""
        widgets = dashboard_manager._get_default_widgets_for_role(
            ExecutiveRole.CTO, "dashboard_id"
        )
        
        assert len(widgets) >= 2  # Base widgets + CTO-specific
        assert all(widget["dashboard_id"] == "dashboard_id" for widget in widgets)
        
        # Check for CTO-specific widgets
        widget_names = [widget["name"] for widget in widgets]
        assert "Technology ROI" in widget_names
        assert "AI Initiative Status" in widget_names
    
    def test_get_default_widgets_for_cfo(self, dashboard_manager):
        """Test getting default widgets for CFO role."""
        widgets = dashboard_manager._get_default_widgets_for_role(
            ExecutiveRole.CFO, "dashboard_id"
        )
        
        assert len(widgets) >= 2  # Base widgets + CFO-specific
        
        # Check for CFO-specific widgets
        widget_names = [widget["name"] for widget in widgets]
        assert "Financial Impact" in widget_names
    
    def test_time_range_creation(self):
        """Test TimeRange creation."""
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow()
        
        time_range = TimeRange(start, end)
        
        assert time_range.start == start
        assert time_range.end == end
    
    def test_share_permissions_creation(self):
        """Test SharePermissions creation."""
        permissions = SharePermissions(
            users=["user1", "user2"],
            permission_type="edit",
            expires_in_days=7
        )
        
        assert permissions.users == ["user1", "user2"]
        assert permissions.permission_type == "edit"
        assert permissions.expires_in_days == 7


class TestDashboardData:
    """Test cases for DashboardData class."""
    
    def test_dashboard_data_creation(self):
        """Test DashboardData creation."""
        mock_dashboard = Mock(spec=Dashboard)
        widgets_data = {"widget1": {"name": "Test Widget"}}
        metrics = [Mock(spec=BusinessMetric)]
        
        dashboard_data = DashboardData(mock_dashboard, widgets_data, metrics)
        
        assert dashboard_data.dashboard == mock_dashboard
        assert dashboard_data.widgets_data == widgets_data
        assert dashboard_data.metrics == metrics
        assert isinstance(dashboard_data.last_updated, datetime)


if __name__ == "__main__":
    pytest.main([__file__])