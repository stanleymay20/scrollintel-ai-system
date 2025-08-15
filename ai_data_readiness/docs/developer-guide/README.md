# Developer Guide

This guide provides comprehensive information for developers working with or extending the AI Data Readiness Platform.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Structure](#code-structure)
4. [API Development](#api-development)
5. [Engine Development](#engine-development)
6. [Testing](#testing)
7. [Database Management](#database-management)
8. [Performance Optimization](#performance-optimization)
9. [Security Considerations](#security-considerations)
10. [Contributing Guidelines](#contributing-guidelines)

## Architecture Overview

### System Architecture

The AI Data Readiness Platform follows a microservices architecture with the following core components:

- **API Layer**: FastAPI-based REST and GraphQL APIs
- **Engine Layer**: Core processing engines for quality, bias, and feature analysis
- **Storage Layer**: PostgreSQL database with Redis caching
- **Processing Layer**: Celery-based background task processing
- **Monitoring Layer**: Prometheus metrics and health checks

### Design Principles

- **Modularity**: Each component is independently deployable and testable
- **Scalability**: Horizontal scaling through containerization and orchestration
- **Reliability**: Comprehensive error handling and recovery mechanisms
- **Security**: Defense-in-depth security model with encryption and access controls
- **Observability**: Comprehensive logging, metrics, and tracing

## Development Environment Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Git
- Docker (optional)

### Local Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/ai-data-readiness-platform.git
   cd ai-data-readiness-platform
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r ai_data_readiness_requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env.development
   # Edit .env.development with your settings
   ```

5. **Set Up Database**
   ```bash
   createdb ai_data_readiness_dev
   python init_database.py
   python -m alembic upgrade head
   ```

6. **Run Tests**
   ```bash
   pytest
   ```

### Development Tools

**Code Quality:**
```bash
# Install pre-commit hooks
pre-commit install

# Run linting
flake8 ai_data_readiness/
black ai_data_readiness/
isort ai_data_readiness/

# Type checking
mypy ai_data_readiness/
```

**Testing:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_data_readiness --cov-report=html

# Run specific test file
pytest tests/test_quality_assessment.py

# Run tests with debugging
pytest -s -vv tests/test_quality_assessment.py::test_specific_function
```

## Code Structure

### Project Layout

```
ai_data_readiness/
├── api/                    # API layer
│   ├── routes/            # API route definitions
│   ├── middleware/        # Request/response middleware
│   ├── models/           # Pydantic models
│   └── app.py            # FastAPI application
├── core/                  # Core functionality
│   ├── config.py         # Configuration management
│   ├── database.py       # Database connections
│   ├── exceptions.py     # Custom exceptions
│   └── security.py       # Security utilities
├── engines/              # Processing engines
│   ├── quality_assessment_engine.py
│   ├── bias_analysis_engine.py
│   ├── feature_engineering_engine.py
│   └── compliance_analyzer.py
├── models/               # Database models
│   ├── base_models.py    # Base SQLAlchemy models
│   ├── dataset_models.py # Dataset-related models
│   └── user_models.py    # User and auth models
├── migrations/           # Database migrations
└── tests/               # Test suite
    ├── unit/            # Unit tests
    ├── integration/     # Integration tests
    └── conftest.py      # Test configuration
```

### Coding Standards

**Python Style:**
- Follow PEP 8 style guide
- Use type hints for all functions
- Maximum line length: 88 characters (Black default)
- Use descriptive variable and function names

**Documentation:**
- All public functions must have docstrings
- Use Google-style docstrings
- Include type information and examples

**Example:**
```python
def assess_data_quality(
    dataset_id: str,
    quality_config: QualityConfig,
    timeout: int = 300
) -> QualityReport:
    """Assess the quality of a dataset.
    
    Args:
        dataset_id: Unique identifier for the dataset
        quality_config: Configuration for quality assessment
        timeout: Maximum time to spend on assessment in seconds
        
    Returns:
        QualityReport containing assessment results
        
    Raises:
        DatasetNotFoundError: If dataset doesn't exist
        QualityAssessmentError: If assessment fails
        
    Example:
        >>> config = QualityConfig(completeness_threshold=0.95)
        >>> report = assess_data_quality("dataset-123", config)
        >>> print(f"Quality score: {report.overall_score}")
    """
    # Implementation here
    pass
```

## API Development

### Adding New Endpoints

1. **Define Pydantic Models**
   ```python
   # api/models/requests.py
   from pydantic import BaseModel, Field
   from typing import Optional, List
   
   class DatasetUploadRequest(BaseModel):
       name: str = Field(..., description="Dataset name")
       description: Optional[str] = Field(None, description="Dataset description")
       tags: List[str] = Field(default_factory=list, description="Dataset tags")
   ```

2. **Create Route Handler**
   ```python
   # api/routes/datasets.py
   from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
   from ..models.requests import DatasetUploadRequest
   from ..models.responses import DatasetResponse
   
   router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])
   
   @router.post("/upload", response_model=DatasetResponse)
   async def upload_dataset(
       file: UploadFile = File(...),
       request: DatasetUploadRequest = Depends(),
       current_user: User = Depends(get_current_user)
   ):
       """Upload a new dataset for analysis."""
       try:
           # Process upload
           dataset = await process_dataset_upload(file, request, current_user)
           return DatasetResponse.from_orm(dataset)
       except Exception as e:
           raise HTTPException(status_code=400, detail=str(e))
   ```

3. **Add to Main App**
   ```python
   # api/app.py
   from .routes import datasets
   
   app.include_router(datasets.router)
   ```

### GraphQL Development

1. **Define GraphQL Types**
   ```python
   # api/graphql/types.py
   import strawberry
   from typing import List, Optional
   
   @strawberry.type
   class Dataset:
       id: str
       name: str
       description: Optional[str]
       quality_score: float
       created_at: str
   ```

2. **Create Resolvers**
   ```python
   # api/graphql/resolvers.py
   @strawberry.type
   class Query:
       @strawberry.field
       async def datasets(
           self,
           limit: int = 20,
           offset: int = 0
       ) -> List[Dataset]:
           """Get list of datasets."""
           return await get_datasets(limit=limit, offset=offset)
   ```

## Engine Development

### Creating New Engines

1. **Base Engine Class**
   ```python
   # engines/base_engine.py
   from abc import ABC, abstractmethod
   from typing import Any, Dict
   
   class BaseEngine(ABC):
       """Base class for all processing engines."""
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.logger = logging.getLogger(self.__class__.__name__)
       
       @abstractmethod
       async def process(self, data: Any) -> Any:
           """Process data and return results."""
           pass
       
       def validate_input(self, data: Any) -> bool:
           """Validate input data."""
           return True
   ```

2. **Implement Specific Engine**
   ```python
   # engines/custom_engine.py
   from .base_engine import BaseEngine
   from ..models.engine_models import CustomEngineResult
   
   class CustomEngine(BaseEngine):
       """Custom processing engine."""
       
       async def process(self, dataset_id: str) -> CustomEngineResult:
           """Process dataset with custom logic."""
           if not self.validate_input(dataset_id):
               raise ValueError("Invalid dataset ID")
           
           try:
               # Custom processing logic
               result = await self._perform_analysis(dataset_id)
               return CustomEngineResult(**result)
           except Exception as e:
               self.logger.error(f"Processing failed: {e}")
               raise
       
       async def _perform_analysis(self, dataset_id: str) -> Dict[str, Any]:
           """Perform the actual analysis."""
           # Implementation here
           pass
   ```

### Engine Registration

```python
# core/engine_registry.py
from typing import Dict, Type
from .engines.base_engine import BaseEngine

class EngineRegistry:
    """Registry for processing engines."""
    
    _engines: Dict[str, Type[BaseEngine]] = {}
    
    @classmethod
    def register(cls, name: str, engine_class: Type[BaseEngine]):
        """Register an engine."""
        cls._engines[name] = engine_class
    
    @classmethod
    def get_engine(cls, name: str) -> Type[BaseEngine]:
        """Get an engine by name."""
        if name not in cls._engines:
            raise ValueError(f"Unknown engine: {name}")
        return cls._engines[name]

# Usage
from .engines.custom_engine import CustomEngine
EngineRegistry.register("custom", CustomEngine)
```

## Testing

### Test Structure

```python
# tests/unit/test_quality_assessment.py
import pytest
from unittest.mock import Mock, patch
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine

class TestQualityAssessmentEngine:
    """Test suite for QualityAssessmentEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing."""
        config = {"completeness_threshold": 0.95}
        return QualityAssessmentEngine(config)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return {
            "id": "test-dataset",
            "data": [[1, 2, 3], [4, None, 6], [7, 8, 9]],
            "columns": ["col1", "col2", "col3"]
        }
    
    async def test_assess_completeness(self, engine, sample_dataset):
        """Test completeness assessment."""
        result = await engine.assess_completeness(sample_dataset)
        
        assert result.overall_score == pytest.approx(0.889, rel=1e-3)
        assert "col2" in result.issues
        assert result.issues["col2"]["missing_count"] == 1
    
    async def test_invalid_dataset_raises_error(self, engine):
        """Test that invalid dataset raises appropriate error."""
        with pytest.raises(ValueError, match="Invalid dataset"):
            await engine.process(None)
    
    @patch('ai_data_readiness.engines.quality_assessment_engine.calculate_statistics')
    async def test_statistics_calculation_called(self, mock_calc, engine, sample_dataset):
        """Test that statistics calculation is called."""
        mock_calc.return_value = {"mean": 5.0, "std": 2.0}
        
        await engine.process(sample_dataset)
        
        mock_calc.assert_called_once()
```

### Integration Tests

```python
# tests/integration/test_api_integration.py
import pytest
from httpx import AsyncClient
from ai_data_readiness.api.app import app

@pytest.mark.asyncio
class TestDatasetAPI:
    """Integration tests for dataset API."""
    
    async def test_upload_dataset_success(self, client: AsyncClient, auth_headers):
        """Test successful dataset upload."""
        files = {"file": ("test.csv", "col1,col2\n1,2\n3,4", "text/csv")}
        data = {"name": "Test Dataset", "description": "Test description"}
        
        response = await client.post(
            "/api/v1/datasets/upload",
            files=files,
            data=data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        result = response.json()
        assert result["name"] == "Test Dataset"
        assert "id" in result
    
    async def test_get_quality_report(self, client: AsyncClient, auth_headers, sample_dataset_id):
        """Test getting quality report for dataset."""
        response = await client.get(
            f"/api/v1/datasets/{sample_dataset_id}/quality",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "overall_score" in result
        assert "completeness_score" in result
```

### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from httpx import AsyncClient
from ai_data_readiness.api.app import app
from ai_data_readiness.core.database import get_db
from ai_data_readiness.models.database import Base

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def auth_headers(test_user_token):
    """Create authentication headers."""
    return {"Authorization": f"Bearer {test_user_token}"}

@pytest.fixture(scope="function")
async def db_session():
    """Create database session for testing."""
    # Setup test database
    engine = create_test_engine()
    Base.metadata.create_all(engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
```

## Database Management

### Creating Migrations

```bash
# Create new migration
python -m alembic revision --autogenerate -m "Add new table"

# Apply migrations
python -m alembic upgrade head

# Downgrade migration
python -m alembic downgrade -1
```

### Model Development

```python
# models/dataset_models.py
from sqlalchemy import Column, String, DateTime, Float, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from .base_models import BaseModel
import uuid

class Dataset(BaseModel):
    """Dataset model."""
    
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(500))
    schema_info = Column(JSON)
    quality_score = Column(Float)
    ai_readiness_score = Column(Float)
    metadata = Column(JSON)
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}')>"
```

## Performance Optimization

### Database Optimization

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# Optimize queries
from sqlalchemy.orm import selectinload

# Eager loading
datasets = session.query(Dataset).options(
    selectinload(Dataset.quality_reports)
).all()

# Batch operations
session.bulk_insert_mappings(Dataset, dataset_data)
session.commit()
```

### Caching

```python
# Redis caching decorator
import redis
import json
from functools import wraps

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute and cache
            result = await func(*args, **kwargs)
            redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result, default=str)
            )
            return result
        return wrapper
    return decorator
```

## Security Considerations

### Input Validation

```python
from pydantic import BaseModel, validator
import re

class DatasetRequest(BaseModel):
    name: str
    description: str = None
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\-\s]+$', v):
            raise ValueError('Name contains invalid characters')
        if len(v) > 255:
            raise ValueError('Name too long')
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if v and len(v) > 1000:
            raise ValueError('Description too long')
        return v
```

### Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    """Get current authenticated user."""
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return await get_user_by_id(user_id)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
```

## Contributing Guidelines

### Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ai-data-readiness-platform.git
   cd ai-data-readiness-platform
   git remote add upstream https://github.com/original-org/ai-data-readiness-platform.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Changes**
   ```bash
   pytest
   flake8 ai_data_readiness/
   black --check ai_data_readiness/
   mypy ai_data_readiness/
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure CI passes

### Code Review Process

- All changes require review from at least one maintainer
- Address feedback promptly
- Keep pull requests focused and small
- Update documentation for user-facing changes

### Release Process

1. **Version Bumping**
   ```bash
   # Update version in setup.py and __init__.py
   git tag v1.2.0
   git push origin v1.2.0
   ```

2. **Release Notes**
   - Document new features
   - List bug fixes
   - Note breaking changes
   - Include migration instructions

This developer guide provides the foundation for contributing to and extending the AI Data Readiness Platform. Follow these guidelines to ensure consistent, high-quality code.