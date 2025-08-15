#!/usr/bin/env python3
"""
Create database migration for Advanced Analytics Dashboard System.
"""
import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from scrollintel.models.database import get_database_url
from scrollintel.models.dashboard_models import Base


def create_dashboard_migration():
    """Create Alembic migration for dashboard models."""
    try:
        # Get database URL
        database_url = get_database_url()
        print(f"Using database: {database_url}")
        
        # Create engine
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("Database connection successful")
        
        # Create Alembic configuration
        alembic_cfg = Config("alembic.ini")
        
        # Generate migration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        migration_message = f"add_dashboard_tables_{timestamp}"
        
        print(f"Creating migration: {migration_message}")
        command.revision(
            alembic_cfg,
            message=migration_message,
            autogenerate=True
        )
        
        print("Migration created successfully!")
        print("To apply the migration, run: alembic upgrade head")
        
    except Exception as e:
        print(f"Error creating migration: {e}")
        sys.exit(1)


def apply_migration():
    """Apply the migration to the database."""
    try:
        alembic_cfg = Config("alembic.ini")
        
        print("Applying migration...")
        command.upgrade(alembic_cfg, "head")
        
        print("Migration applied successfully!")
        
    except Exception as e:
        print(f"Error applying migration: {e}")
        sys.exit(1)


def create_sample_data():
    """Create sample dashboard data for testing."""
    try:
        from sqlalchemy.orm import sessionmaker
        from scrollintel.models.database import get_db_session
        from scrollintel.models.dashboard_models import (
            Dashboard, Widget, DashboardTemplate, BusinessMetric,
            DashboardType, ExecutiveRole, WidgetType
        )
        from scrollintel.core.dashboard_templates import DashboardTemplates
        
        print("Creating sample dashboard data...")
        
        with get_db_session() as db:
            # Create sample dashboard templates
            templates = DashboardTemplates.get_all_templates()
            
            for template_key, template_config in templates.items():
                existing_template = db.query(DashboardTemplate).filter(
                    DashboardTemplate.name == template_config["name"]
                ).first()
                
                if not existing_template:
                    template = DashboardTemplate(
                        name=template_config["name"],
                        category=template_config["category"],
                        role=template_config["role"],
                        description=template_config["description"],
                        template_config=template_config,
                        created_by="system"
                    )
                    db.add(template)
                    print(f"Created template: {template_config['name']}")
            
            # Create sample executive dashboard
            existing_dashboard = db.query(Dashboard).filter(
                Dashboard.name == "Sample CTO Dashboard"
            ).first()
            
            if not existing_dashboard:
                dashboard = Dashboard(
                    name="Sample CTO Dashboard",
                    type=DashboardType.EXECUTIVE.value,
                    owner_id="sample_user",
                    role=ExecutiveRole.CTO.value,
                    description="Sample CTO executive dashboard",
                    config={
                        "layout": {"grid_columns": 12, "grid_rows": 8},
                        "theme": "executive",
                        "auto_refresh": True,
                        "refresh_interval": 300
                    }
                )
                db.add(dashboard)
                db.flush()
                
                # Add sample widgets
                widgets = [
                    {
                        "name": "Technology ROI",
                        "type": WidgetType.KPI.value,
                        "position_x": 0,
                        "position_y": 0,
                        "width": 12,
                        "height": 2,
                        "config": {"metrics": ["tech_roi", "ai_investment_return"]},
                        "data_source": "roi_calculator"
                    },
                    {
                        "name": "AI Initiative Performance",
                        "type": WidgetType.CHART.value,
                        "position_x": 0,
                        "position_y": 2,
                        "width": 8,
                        "height": 3,
                        "config": {"chart_type": "line", "time_range": "30d"},
                        "data_source": "ai_projects"
                    },
                    {
                        "name": "System Health",
                        "type": WidgetType.METRIC.value,
                        "position_x": 8,
                        "position_y": 2,
                        "width": 4,
                        "height": 3,
                        "config": {"metric": "system_uptime"},
                        "data_source": "monitoring"
                    }
                ]
                
                for widget_config in widgets:
                    widget = Widget(
                        dashboard_id=dashboard.id,
                        **widget_config
                    )
                    db.add(widget)
                
                print("Created sample CTO dashboard")
            
            # Create sample metrics
            sample_metrics = [
                {
                    "name": "tech_roi",
                    "category": "financial",
                    "value": 18.5,
                    "unit": "%",
                    "source": "roi_calculator",
                    "context": {"trend": "up"}
                },
                {
                    "name": "ai_investment_return",
                    "category": "financial",
                    "value": 250000,
                    "unit": "USD",
                    "source": "financial_system",
                    "context": {"trend": "up"}
                },
                {
                    "name": "system_uptime",
                    "category": "operational",
                    "value": 99.8,
                    "unit": "%",
                    "source": "monitoring",
                    "context": {"trend": "neutral"}
                },
                {
                    "name": "deployment_frequency",
                    "category": "operational",
                    "value": 12,
                    "unit": "per week",
                    "source": "ci_cd",
                    "context": {"trend": "up"}
                }
            ]
            
            for metric_data in sample_metrics:
                existing_metric = db.query(BusinessMetric).filter(
                    BusinessMetric.name == metric_data["name"]
                ).first()
                
                if not existing_metric:
                    metric = BusinessMetric(**metric_data)
                    db.add(metric)
            
            db.commit()
            print("Sample data created successfully!")
            
    except Exception as e:
        print(f"Error creating sample data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dashboard migration utility")
    parser.add_argument("--create", action="store_true", help="Create migration")
    parser.add_argument("--apply", action="store_true", help="Apply migration")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data")
    parser.add_argument("--all", action="store_true", help="Create, apply migration and add sample data")
    
    args = parser.parse_args()
    
    if args.all:
        create_dashboard_migration()
        apply_migration()
        create_sample_data()
    elif args.create:
        create_dashboard_migration()
    elif args.apply:
        apply_migration()
    elif args.sample_data:
        create_sample_data()
    else:
        parser.print_help()