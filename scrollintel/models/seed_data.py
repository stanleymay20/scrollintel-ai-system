"""
Database seed data for ScrollIntel system.
"""

from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import uuid

from .database import User, Agent, Dataset, Dashboard
from ..core.interfaces import AgentType, AgentStatus, UserRole


def create_default_users(session: Session) -> List[User]:
    """Create default users for the system."""
    users = []
    
    # Admin user
    admin_user = User(
        id=uuid.uuid4(),
        email="admin@scrollintel.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3L3jHZZZZe",  # password: admin123
        role=UserRole.ADMIN,
        permissions=["*"],
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    users.append(admin_user)
    
    # Analyst user
    analyst_user = User(
        id=uuid.uuid4(),
        email="analyst@scrollintel.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3L3jHZZZZe",  # password: analyst123
        role=UserRole.ANALYST,
        permissions=["read:datasets", "write:datasets", "read:models", "write:models", "read:dashboards", "write:dashboards"],
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    users.append(analyst_user)
    
    # Viewer user
    viewer_user = User(
        id=uuid.uuid4(),
        email="viewer@scrollintel.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3L3jHZZZZe",  # password: viewer123
        role=UserRole.VIEWER,
        permissions=["read:dashboards"],
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    users.append(viewer_user)
    
    # Add users to session
    for user in users:
        session.add(user)
    
    return users


def create_default_agents(session: Session) -> List[Agent]:
    """Create default AI agents for the system."""
    agents = []
    
    # ScrollCTOAgent
    cto_agent = Agent(
        id=uuid.uuid4(),
        name="ScrollCTOAgent",
        agent_type=AgentType.CTO,
        capabilities=[
            "technical_architecture",
            "stack_planning",
            "scaling_strategy",
            "technology_comparison",
            "cost_analysis"
        ],
        status=AgentStatus.ACTIVE,
        configuration={
            "ai_model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.7,
            "specialization": "technical_leadership"
        },
        version="1.0.0",
        description="AI agent specialized in technical architecture and CTO-level decision making",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    agents.append(cto_agent)
    
    # ScrollDataScientist
    data_scientist_agent = Agent(
        id=uuid.uuid4(),
        name="ScrollDataScientist",
        agent_type=AgentType.DATA_SCIENTIST,
        capabilities=[
            "exploratory_data_analysis",
            "statistical_modeling",
            "hypothesis_testing",
            "data_preprocessing",
            "feature_engineering"
        ],
        status=AgentStatus.ACTIVE,
        configuration={
            "ai_model": "claude-3-sonnet-20240229",
            "max_tokens": 4000,
            "temperature": 0.3,
            "specialization": "data_science"
        },
        version="1.0.0",
        description="AI agent specialized in data science and statistical analysis",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    agents.append(data_scientist_agent)
    
    # ScrollMLEngineer
    ml_engineer_agent = Agent(
        id=uuid.uuid4(),
        name="ScrollMLEngineer",
        agent_type=AgentType.ML_ENGINEER,
        capabilities=[
            "ml_pipeline_setup",
            "model_training",
            "model_deployment",
            "model_monitoring",
            "mlops_automation"
        ],
        status=AgentStatus.ACTIVE,
        configuration={
            "ai_model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.5,
            "specialization": "ml_engineering"
        },
        version="1.0.0",
        description="AI agent specialized in ML engineering and MLOps",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    agents.append(ml_engineer_agent)
    
    # ScrollAIEngineer
    ai_engineer_agent = Agent(
        id=uuid.uuid4(),
        name="ScrollAIEngineer",
        agent_type=AgentType.AI_ENGINEER,
        capabilities=[
            "rag_implementation",
            "vector_operations",
            "embedding_generation",
            "langchain_workflows",
            "llm_integration"
        ],
        status=AgentStatus.ACTIVE,
        configuration={
            "ai_model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.4,
            "specialization": "ai_engineering"
        },
        version="1.0.0",
        description="AI agent specialized in AI engineering and LLM integration",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    agents.append(ai_engineer_agent)
    
    # ScrollAnalyst
    analyst_agent = Agent(
        id=uuid.uuid4(),
        name="ScrollAnalyst",
        agent_type=AgentType.ANALYST,
        capabilities=[
            "sql_generation",
            "kpi_calculation",
            "business_insights",
            "report_generation",
            "trend_analysis"
        ],
        status=AgentStatus.ACTIVE,
        configuration={
            "ai_model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.3,
            "specialization": "business_intelligence"
        },
        version="1.0.0",
        description="AI agent specialized in business analysis and KPI generation",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    agents.append(analyst_agent)
    
    # ScrollBI
    bi_agent = Agent(
        id=uuid.uuid4(),
        name="ScrollBI",
        agent_type=AgentType.BI_DEVELOPER,
        capabilities=[
            "dashboard_creation",
            "visualization_generation",
            "real_time_updates",
            "alert_management",
            "chart_optimization"
        ],
        status=AgentStatus.ACTIVE,
        configuration={
            "ai_model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.6,
            "specialization": "business_intelligence"
        },
        version="1.0.0",
        description="AI agent specialized in dashboard creation and business intelligence",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    agents.append(bi_agent)
    
    # Add agents to session
    for agent in agents:
        session.add(agent)
    
    return agents


def create_sample_datasets(session: Session) -> List[Dataset]:
    """Create sample datasets for demonstration."""
    datasets = []
    
    # Sales dataset
    sales_dataset = Dataset(
        id=uuid.uuid4(),
        name="Sales Data",
        source_type="csv",
        data_schema={
            "date": "datetime",
            "product_id": "string",
            "product_name": "string",
            "category": "string",
            "quantity": "integer",
            "unit_price": "float",
            "total_amount": "float",
            "customer_id": "string",
            "region": "string"
        },
        row_count=10000,
        file_path="/data/sales_data.csv",
        dataset_metadata={
            "description": "Historical sales data for the past 2 years",
            "date_range": "2022-01-01 to 2024-01-01",
            "update_frequency": "daily"
        },
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    datasets.append(sales_dataset)
    
    # Customer dataset
    customer_dataset = Dataset(
        id=uuid.uuid4(),
        name="Customer Data",
        source_type="json",
        data_schema={
            "customer_id": "string",
            "first_name": "string",
            "last_name": "string",
            "email": "string",
            "phone": "string",
            "address": "string",
            "city": "string",
            "state": "string",
            "zip_code": "string",
            "registration_date": "datetime",
            "customer_segment": "string"
        },
        row_count=5000,
        file_path="/data/customers.json",
        dataset_metadata={
            "description": "Customer information and demographics",
            "privacy_level": "high",
            "update_frequency": "weekly"
        },
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    datasets.append(customer_dataset)
    
    # Product dataset
    product_dataset = Dataset(
        id=uuid.uuid4(),
        name="Product Catalog",
        source_type="xlsx",
        data_schema={
            "product_id": "string",
            "product_name": "string",
            "category": "string",
            "subcategory": "string",
            "brand": "string",
            "unit_price": "float",
            "cost": "float",
            "margin": "float",
            "stock_quantity": "integer",
            "supplier": "string"
        },
        row_count=1500,
        file_path="/data/products.xlsx",
        dataset_metadata={
            "description": "Complete product catalog with pricing and inventory",
            "update_frequency": "daily"
        },
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    datasets.append(product_dataset)
    
    # Add datasets to session
    for dataset in datasets:
        session.add(dataset)
    
    return datasets


def create_sample_dashboards(session: Session, users: List[User]) -> List[Dashboard]:
    """Create sample dashboards."""
    dashboards = []
    
    if not users:
        return dashboards
    
    admin_user = next((u for u in users if u.role == UserRole.ADMIN), users[0])
    
    # Sales dashboard
    sales_dashboard = Dashboard(
        id=uuid.uuid4(),
        name="Sales Performance Dashboard",
        user_id=admin_user.id,
        config={
            "theme": "dark",
            "auto_refresh": True,
            "layout": "grid"
        },
        charts=[
            {
                "id": "sales_trend",
                "type": "line",
                "title": "Sales Trend",
                "data_source": "sales_data",
                "x_axis": "date",
                "y_axis": "total_amount"
            },
            {
                "id": "top_products",
                "type": "bar",
                "title": "Top Products",
                "data_source": "sales_data",
                "x_axis": "product_name",
                "y_axis": "quantity"
            },
            {
                "id": "regional_sales",
                "type": "pie",
                "title": "Sales by Region",
                "data_source": "sales_data",
                "category": "region",
                "value": "total_amount"
            }
        ],
        refresh_interval_minutes=300,
        is_public=False,
        is_active=True,

        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    dashboards.append(sales_dashboard)
    
    # Customer analytics dashboard
    customer_dashboard = Dashboard(
        id=uuid.uuid4(),
        name="Customer Analytics",
        user_id=admin_user.id,
        config={
            "theme": "light",
            "auto_refresh": True,
            "layout": "flex"
        },
        charts=[
            {
                "id": "customer_segments",
                "type": "donut",
                "title": "Customer Segments",
                "data_source": "customer_data",
                "category": "customer_segment",
                "value": "count"
            },
            {
                "id": "registration_trend",
                "type": "area",
                "title": "New Customer Registrations",
                "data_source": "customer_data",
                "x_axis": "registration_date",
                "y_axis": "count"
            }
        ],
        refresh_interval_minutes=600,
        is_public=True,
        is_active=True,
        tags=["customers", "analytics", "segments"],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    dashboards.append(customer_dashboard)
    
    # Add dashboards to session
    for dashboard in dashboards:
        session.add(dashboard)
    
    return dashboards


def seed_database(session: Session) -> Dict[str, Any]:
    """Seed the database with initial data."""
    try:
        # Create default users
        users = create_default_users(session)
        
        # Create default agents
        agents = create_default_agents(session)
        
        # Create sample datasets
        datasets = create_sample_datasets(session)
        
        # Create sample dashboards
        dashboards = create_sample_dashboards(session, users)
        
        # Commit all changes
        session.commit()
        
        return {
            "success": True,
            "message": "Database seeded successfully",
            "data": {
                "users_created": len(users),
                "agents_created": len(agents),
                "datasets_created": len(datasets),
                "dashboards_created": len(dashboards)
            }
        }
        
    except Exception as e:
        session.rollback()
        return {
            "success": False,
            "message": f"Failed to seed database: {str(e)}",
            "data": None
        }


def clear_seed_data(session: Session) -> Dict[str, Any]:
    """Clear all seed data from the database."""
    try:
        # Delete in reverse order of dependencies
        session.query(Dashboard).delete()
        session.query(Dataset).delete()
        session.query(Agent).delete()
        session.query(User).delete()
        
        session.commit()
        
        return {
            "success": True,
            "message": "Seed data cleared successfully"
        }
        
    except Exception as e:
        session.rollback()
        return {
            "success": False,
            "message": f"Failed to clear seed data: {str(e)}"
        }