"""
Unit tests for ScrollIntel data models and validation.
"""

import pytest
from datetime import datetime
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from pydantic import ValidationError

from scrollintel.models.database import Base, User, Agent, Dataset, MLModel, Dashboard, AgentRequest, AgentResponse, AuditLog
from scrollintel.models.schemas import (
    UserCreate, UserUpdate, UserResponse,
    AgentCreate, AgentUpdate, AgentResponse as AgentResponseSchema,
    DatasetCreate, DatasetUpdate, DatasetResponse,
    MLModelCreate, MLModelUpdate, MLModelResponse,
    DashboardCreate, DashboardUpdate, DashboardResponse,
    AgentRequestCreate, AgentResponseCreate,
    AuditLogCreate
)
from scrollintel.core.interfaces import AgentType, AgentStatus, ResponseStatus, UserRole


@pytest.fixture
def db_session():
    """Create a test database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


class TestUserModel:
    """Test User model and schemas."""
    
    def test_user_creation(self, db_session):
        """Test creating a user in the database."""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.ANALYST,
            permissions=["read:data", "write:data"],
            is_active=True
        )
        
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.role == UserRole.ANALYST
        assert user.permissions == ["read:data", "write:data"]
        assert user.is_active is True
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_create_schema_validation(self):
        """Test UserCreate schema validation."""
        # Valid user creation
        valid_user = UserCreate(
            email="test@example.com",
            password="SecurePass123",
            role=UserRole.ANALYST,
            permissions=["read:data"]
        )
        assert valid_user.email == "test@example.com"
        assert valid_user.password == "SecurePass123"
        
        # Invalid email
        with pytest.raises(ValidationError):
            UserCreate(
                email="invalid-email",
                password="SecurePass123"
            )
        
        # Invalid password (too short)
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                password="short"
            )
        
        # Invalid password (no uppercase)
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                password="lowercase123"
            )
        
        # Invalid password (no digit)
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                password="NoDigitsHere"
            )
    
    def test_user_update_schema(self):
        """Test UserUpdate schema."""
        update_data = UserUpdate(
            email="newemail@example.com",
            role=UserRole.ADMIN,
            is_active=False
        )
        
        assert update_data.email == "newemail@example.com"
        assert update_data.role == UserRole.ADMIN
        assert update_data.is_active is False
    
    def test_user_response_schema(self):
        """Test UserResponse schema."""
        user_data = {
            "id": uuid4(),
            "email": "test@example.com",
            "role": UserRole.VIEWER,
            "permissions": ["read:dashboards"],
            "is_active": True,
            "last_login": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        user_response = UserResponse(**user_data)
        assert user_response.email == "test@example.com"
        assert user_response.role == UserRole.VIEWER


class TestAgentModel:
    """Test Agent model and schemas."""
    
    def test_agent_creation(self, db_session):
        """Test creating an agent in the database."""
        agent = Agent(
            name="TestAgent",
            agent_type=AgentType.DATA_SCIENTIST,
            capabilities=["data_analysis", "visualization"],
            status=AgentStatus.ACTIVE,
            configuration={"model": "gpt-4", "temperature": 0.7},
            description="Test agent for data science tasks"
        )
        
        db_session.add(agent)
        db_session.commit()
        
        assert agent.id is not None
        assert agent.name == "TestAgent"
        assert agent.agent_type == AgentType.DATA_SCIENTIST
        assert agent.capabilities == ["data_analysis", "visualization"]
        assert agent.status == AgentStatus.ACTIVE
        assert agent.configuration == {"model": "gpt-4", "temperature": 0.7}
    
    def test_agent_create_schema(self):
        """Test AgentCreate schema validation."""
        valid_agent = AgentCreate(
            name="ScrollCTOAgent",
            agent_type=AgentType.CTO,
            capabilities=["architecture", "scaling"],
            configuration={"model": "gpt-4"},
            description="CTO agent"
        )
        
        assert valid_agent.name == "ScrollCTOAgent"
        assert valid_agent.agent_type == AgentType.CTO
        
        # Test name validation
        with pytest.raises(ValidationError):
            AgentCreate(
                name="",  # Empty name
                agent_type=AgentType.CTO
            )


class TestDatasetModel:
    """Test Dataset model and schemas."""
    
    def test_dataset_creation(self, db_session):
        """Test creating a dataset in the database."""
        dataset = Dataset(
            name="Sales Data",
            source_type="csv",
            data_schema={"date": "datetime", "amount": "float"},
            row_count=1000,
            file_path="/data/sales.csv",
            dataset_metadata={"description": "Sales data for 2024"}
        )
        
        db_session.add(dataset)
        db_session.commit()
        
        assert dataset.id is not None
        assert dataset.name == "Sales Data"
        assert dataset.source_type == "csv"
        assert dataset.row_count == 1000
        assert dataset.is_active is True
    
    def test_dataset_create_schema(self):
        """Test DatasetCreate schema validation."""
        valid_dataset = DatasetCreate(
            name="Test Dataset",
            source_type="json",
            data_schema={"id": "string", "value": "float"},
            row_count=500
        )
        
        assert valid_dataset.name == "Test Dataset"
        assert valid_dataset.source_type == "json"
        
        # Test invalid source type
        with pytest.raises(ValidationError):
            DatasetCreate(
                name="Test Dataset",
                source_type="invalid_type"
            )
        
        # Test negative row count
        with pytest.raises(ValidationError):
            DatasetCreate(
                name="Test Dataset",
                source_type="csv",
                row_count=-1
            )


class TestMLModelModel:
    """Test MLModel model and schemas."""
    
    def test_mlmodel_creation(self, db_session):
        """Test creating an ML model in the database."""
        # First create a dataset
        dataset = Dataset(
            name="Training Data",
            source_type="csv",
            schema={"feature1": "float", "target": "float"}
        )
        db_session.add(dataset)
        db_session.commit()
        
        # Create ML model
        ml_model = MLModel(
            name="Random Forest Model",
            algorithm="random_forest",
            dataset_id=dataset.id,
            parameters={"n_estimators": 100, "max_depth": 10},
            metrics={"accuracy": 0.95, "f1_score": 0.92},
            model_path="/models/rf_model.pkl",
            training_duration=120.5
        )
        
        db_session.add(ml_model)
        db_session.commit()
        
        assert ml_model.id is not None
        assert ml_model.name == "Random Forest Model"
        assert ml_model.algorithm == "random_forest"
        assert ml_model.dataset_id == dataset.id
        assert ml_model.metrics["accuracy"] == 0.95
        assert ml_model.is_deployed is False
    
    def test_mlmodel_create_schema(self):
        """Test MLModelCreate schema validation."""
        dataset_id = uuid4()
        
        valid_model = MLModelCreate(
            name="Test Model",
            algorithm="xgboost",
            dataset_id=dataset_id,
            model_path="/models/test.pkl",
            parameters={"learning_rate": 0.1},
            metrics={"rmse": 0.05}
        )
        
        assert valid_model.name == "Test Model"
        assert valid_model.algorithm == "xgboost"
        assert valid_model.dataset_id == dataset_id


class TestDashboardModel:
    """Test Dashboard model and schemas."""
    
    def test_dashboard_creation(self, db_session):
        """Test creating a dashboard in the database."""
        # First create a user
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.ANALYST
        )
        db_session.add(user)
        db_session.commit()
        
        # Create dashboard
        dashboard = Dashboard(
            name="Sales Dashboard",
            user_id=user.id,
            config={"theme": "dark", "auto_refresh": True},
            charts=[{"type": "line", "title": "Sales Trend"}],
            refresh_interval=300,
            is_public=False,
            tags=["sales", "analytics"]
        )
        
        db_session.add(dashboard)
        db_session.commit()
        
        assert dashboard.id is not None
        assert dashboard.name == "Sales Dashboard"
        assert dashboard.user_id == user.id
        assert dashboard.refresh_interval == 300
        assert dashboard.tags == ["sales", "analytics"]
    
    def test_dashboard_create_schema(self):
        """Test DashboardCreate schema validation."""
        valid_dashboard = DashboardCreate(
            name="Test Dashboard",
            config={"theme": "light"},
            charts=[{"type": "bar", "title": "Test Chart"}],
            refresh_interval=600,
            tags=["test"]
        )
        
        assert valid_dashboard.name == "Test Dashboard"
        assert valid_dashboard.refresh_interval == 600
        
        # Test invalid refresh interval (too short)
        with pytest.raises(ValidationError):
            DashboardCreate(
                name="Test Dashboard",
                refresh_interval=10  # Less than 30 seconds
            )
        
        # Test invalid refresh interval (too long)
        with pytest.raises(ValidationError):
            DashboardCreate(
                name="Test Dashboard",
                refresh_interval=90000  # More than 24 hours
            )


class TestAgentRequestResponseModels:
    """Test AgentRequest and AgentResponse models."""
    
    def test_agent_request_creation(self, db_session):
        """Test creating an agent request in the database."""
        # Create user and agent first
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.ANALYST
        )
        agent = Agent(
            name="TestAgent",
            agent_type=AgentType.DATA_SCIENTIST,
            status=AgentStatus.ACTIVE
        )
        
        db_session.add(user)
        db_session.add(agent)
        db_session.commit()
        
        # Create agent request
        request = AgentRequest(
            user_id=user.id,
            agent_id=agent.id,
            prompt="Analyze the sales data",
            context={"dataset": "sales_2024"},
            priority=2
        )
        
        db_session.add(request)
        db_session.commit()
        
        assert request.id is not None
        assert request.user_id == user.id
        assert request.agent_id == agent.id
        assert request.prompt == "Analyze the sales data"
        assert request.priority == 2
    
    def test_agent_response_creation(self, db_session):
        """Test creating an agent response in the database."""
        # Create user, agent, and request first
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.ANALYST
        )
        agent = Agent(
            name="TestAgent",
            agent_type=AgentType.DATA_SCIENTIST,
            status=AgentStatus.ACTIVE
        )
        
        db_session.add(user)
        db_session.add(agent)
        db_session.commit()
        
        request = AgentRequest(
            user_id=user.id,
            agent_id=agent.id,
            prompt="Test prompt"
        )
        
        db_session.add(request)
        db_session.commit()
        
        # Create agent response
        response = AgentResponse(
            request_id=request.id,
            agent_id=agent.id,
            content="Analysis complete. Found 3 key trends.",
            artifacts=["chart1.png", "report.pdf"],
            execution_time=45.2,
            status=ResponseStatus.SUCCESS
        )
        
        db_session.add(response)
        db_session.commit()
        
        assert response.id is not None
        assert response.request_id == request.id
        assert response.agent_id == agent.id
        assert response.status == ResponseStatus.SUCCESS
        assert response.execution_time == 45.2
    
    def test_agent_request_create_schema(self):
        """Test AgentRequestCreate schema validation."""
        agent_id = uuid4()
        
        valid_request = AgentRequestCreate(
            agent_id=agent_id,
            prompt="Test prompt",
            context={"key": "value"},
            priority=3
        )
        
        assert valid_request.agent_id == agent_id
        assert valid_request.priority == 3
        
        # Test invalid priority
        with pytest.raises(ValidationError):
            AgentRequestCreate(
                agent_id=agent_id,
                prompt="Test prompt",
                priority=11  # Greater than 10
            )
    
    def test_agent_response_create_schema(self):
        """Test AgentResponseCreate schema validation."""
        request_id = uuid4()
        agent_id = uuid4()
        
        valid_response = AgentResponseCreate(
            request_id=request_id,
            agent_id=agent_id,
            content="Response content",
            execution_time=30.5,
            status=ResponseStatus.SUCCESS,
            artifacts=["file1.txt"]
        )
        
        assert valid_response.request_id == request_id
        assert valid_response.execution_time == 30.5
        
        # Test negative execution time
        with pytest.raises(ValidationError):
            AgentResponseCreate(
                request_id=request_id,
                agent_id=agent_id,
                content="Response content",
                execution_time=-5.0,  # Negative time
                status=ResponseStatus.SUCCESS
            )


class TestAuditLogModel:
    """Test AuditLog model and schemas."""
    
    def test_audit_log_creation(self, db_session):
        """Test creating an audit log entry in the database."""
        # Create user first
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.ANALYST
        )
        db_session.add(user)
        db_session.commit()
        
        # Create audit log
        audit_log = AuditLog(
            user_id=user.id,
            action="CREATE",
            resource_type="dataset",
            resource_id="dataset_123",
            details={"name": "New Dataset", "size": 1000},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0...",
            session_id="session_123"
        )
        
        db_session.add(audit_log)
        db_session.commit()
        
        assert audit_log.id is not None
        assert audit_log.user_id == user.id
        assert audit_log.action == "CREATE"
        assert audit_log.resource_type == "dataset"
        assert audit_log.ip_address == "192.168.1.1"
        assert audit_log.timestamp is not None
    
    def test_audit_log_create_schema(self):
        """Test AuditLogCreate schema validation."""
        user_id = uuid4()
        
        valid_audit = AuditLogCreate(
            user_id=user_id,
            action="UPDATE",
            resource_type="dashboard",
            resource_id="dash_456",
            details={"field": "name", "old": "Old Name", "new": "New Name"},
            ip_address="10.0.0.1"
        )
        
        assert valid_audit.user_id == user_id
        assert valid_audit.action == "UPDATE"
        assert valid_audit.resource_type == "dashboard"


class TestModelRelationships:
    """Test relationships between models."""
    
    def test_user_dashboard_relationship(self, db_session):
        """Test User-Dashboard relationship."""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.ANALYST
        )
        db_session.add(user)
        db_session.commit()
        
        dashboard1 = Dashboard(
            name="Dashboard 1",
            user_id=user.id,
            config={}
        )
        dashboard2 = Dashboard(
            name="Dashboard 2",
            user_id=user.id,
            config={}
        )
        
        db_session.add(dashboard1)
        db_session.add(dashboard2)
        db_session.commit()
        
        # Test relationship
        assert len(user.dashboards) == 2
        assert dashboard1.user == user
        assert dashboard2.user == user
    
    def test_dataset_mlmodel_relationship(self, db_session):
        """Test Dataset-MLModel relationship."""
        dataset = Dataset(
            name="Training Data",
            source_type="csv",
            schema={}
        )
        db_session.add(dataset)
        db_session.commit()
        
        model1 = MLModel(
            name="Model 1",
            algorithm="rf",
            dataset_id=dataset.id,
            model_path="/models/model1.pkl"
        )
        model2 = MLModel(
            name="Model 2",
            algorithm="xgb",
            dataset_id=dataset.id,
            model_path="/models/model2.pkl"
        )
        
        db_session.add(model1)
        db_session.add(model2)
        db_session.commit()
        
        # Test relationship
        assert len(dataset.ml_models) == 2
        assert model1.dataset == dataset
        assert model2.dataset == dataset
    
    def test_agent_request_response_relationship(self, db_session):
        """Test Agent-Request-Response relationships."""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.ANALYST
        )
        agent = Agent(
            name="TestAgent",
            agent_type=AgentType.DATA_SCIENTIST,
            status=AgentStatus.ACTIVE
        )
        
        db_session.add(user)
        db_session.add(agent)
        db_session.commit()
        
        request = AgentRequest(
            user_id=user.id,
            agent_id=agent.id,
            prompt="Test prompt"
        )
        db_session.add(request)
        db_session.commit()
        
        response = AgentResponse(
            request_id=request.id,
            agent_id=agent.id,
            content="Response content",
            execution_time=10.0,
            status=ResponseStatus.SUCCESS
        )
        db_session.add(response)
        db_session.commit()
        
        # Test relationships
        assert len(user.agent_requests) == 1
        assert len(agent.agent_requests) == 1
        assert len(agent.agent_responses) == 1
        assert len(request.responses) == 1
        assert response.request == request
        assert response.agent == agent


class TestDatabaseIndexes:
    """Test database indexes and constraints."""
    
    def test_user_email_unique_constraint(self, db_session):
        """Test that user email must be unique."""
        user1 = User(
            email="test@example.com",
            hashed_password="password1",
            role=UserRole.ANALYST
        )
        user2 = User(
            email="test@example.com",  # Same email
            hashed_password="password2",
            role=UserRole.VIEWER
        )
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(Exception):  # Should raise integrity error
            db_session.commit()
    
    def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraints."""
        # Try to create a dashboard without a valid user
        invalid_dashboard = Dashboard(
            name="Invalid Dashboard",
            user_id=uuid4(),  # Non-existent user ID
            config={}
        )
        
        db_session.add(invalid_dashboard)
        with pytest.raises(Exception):  # Should raise foreign key constraint error
            db_session.commit()


if __name__ == "__main__":
    pytest.main([__file__])