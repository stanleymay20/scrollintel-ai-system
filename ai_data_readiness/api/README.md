# AI Data Readiness Platform API

Comprehensive REST and GraphQL API for AI data preparation, quality assessment, and optimization.

## Features

### REST API
- **Authentication**: JWT-based authentication with role-based permissions
- **Dataset Management**: CRUD operations for datasets with metadata
- **Quality Assessment**: Comprehensive data quality evaluation
- **Bias Analysis**: Fairness and bias detection across protected attributes
- **Feature Engineering**: Intelligent feature recommendations
- **Compliance Checking**: GDPR, CCPA, and other regulatory compliance
- **Data Lineage**: Complete data transformation tracking
- **Drift Monitoring**: Real-time data distribution change detection
- **Processing Jobs**: Asynchronous job management

### GraphQL API
- **Complex Queries**: Flexible data querying with relationships
- **Real-time Subscriptions**: Live updates for datasets, jobs, and alerts
- **Batch Operations**: Efficient bulk data operations
- **Analytics**: Advanced analytics and reporting capabilities

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DB_HOST=localhost
export DB_NAME=ai_data_readiness
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### Running the API

```bash
# Development server
python -m ai_data_readiness.api.app

# Production server
uvicorn ai_data_readiness.api.app:app --host 0.0.0.0 --port 8000
```

### API Documentation

- **REST API Docs**: http://localhost:8000/docs
- **GraphQL Playground**: http://localhost:8000/graphql/playground
- **Health Check**: http://localhost:8000/health

## Authentication

### Login
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### Using the Token
```bash
curl -X GET "http://localhost:8000/api/v1/datasets" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## REST API Examples

### Create Dataset
```bash
curl -X POST "http://localhost:8000/api/v1/datasets" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Data",
    "description": "Customer transaction dataset",
    "source": "database",
    "format": "csv",
    "tags": ["customer", "transactions"]
  }'
```

### Assess Quality
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/{dataset_id}/quality" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset_id",
    "dimensions": ["completeness", "accuracy", "consistency"],
    "generate_recommendations": true
  }'
```

### Analyze Bias
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/{dataset_id}/bias" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset_id",
    "protected_attributes": ["gender", "age", "race"],
    "target_column": "outcome"
  }'
```

## GraphQL Examples

### Query Datasets with Quality Reports
```graphql
query GetDatasetsWithQuality {
  datasets(limit: 10) {
    id
    name
    qualityScore
    aiReadinessScore
    status
    isAiReady(threshold: 0.8)
    qualityGrade
  }
}
```

### Complex Query with Relationships
```graphql
query GetDatasetDetails($datasetId: String!) {
  dataset(datasetId: $datasetId) {
    id
    name
    description
    qualityScore
    aiReadinessScore
  }
  
  qualityReport(datasetId: $datasetId) {
    overallScore
    dimensionScores
    issues {
      dimension
      severity
      description
    }
    recommendations {
      type
      priority
      description
    }
  }
  
  biasReport(datasetId: $datasetId) {
    biasMetrics
    fairnessViolations {
      biasType
      severity
      description
    }
  }
  
  lineage(datasetId: $datasetId) {
    sourceDatasets
    downstreamDatasets
    modelsTrained
  }
}
```

### Real-time Subscriptions
```graphql
subscription DatasetUpdates($datasetId: String!) {
  datasetUpdates(datasetId: $datasetId) {
    id
    name
    status
    qualityScore
    aiReadinessScore
  }
}

subscription JobStatusUpdates($jobId: String!) {
  jobStatusUpdates(jobId: $jobId) {
    jobId
    status
    progress
    result
  }
}
```

### Mutations
```graphql
mutation CreateDataset($input: DatasetCreateInput!) {
  createDataset(input: $input) {
    id
    name
    status
  }
}

mutation AssessQuality($input: QualityAssessmentInput!) {
  assessQuality(input: $input) {
    datasetId
    overallScore
    issues {
      dimension
      severity
      description
    }
  }
}
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout

### Datasets
- `GET /api/v1/datasets` - List datasets
- `POST /api/v1/datasets` - Create dataset
- `GET /api/v1/datasets/{id}` - Get dataset
- `PUT /api/v1/datasets/{id}` - Update dataset
- `DELETE /api/v1/datasets/{id}` - Delete dataset
- `POST /api/v1/datasets/upload` - Upload dataset file
- `POST /api/v1/datasets/bulk` - Bulk operations
- `GET /api/v1/datasets/metrics` - System metrics

### Quality Assessment
- `POST /api/v1/datasets/{id}/quality` - Assess quality
- `GET /api/v1/datasets/{id}/quality` - Get quality report
- `GET /api/v1/datasets/{id}/ai-readiness` - Get AI readiness

### Bias Analysis
- `POST /api/v1/datasets/{id}/bias` - Analyze bias
- `GET /api/v1/datasets/{id}/bias` - Get bias report

### Feature Engineering
- `POST /api/v1/datasets/{id}/features` - Get feature recommendations

### Compliance
- `POST /api/v1/datasets/{id}/compliance` - Check compliance

### Data Lineage
- `GET /api/v1/datasets/{id}/lineage` - Get data lineage

### Drift Monitoring
- `POST /api/v1/datasets/{id}/drift` - Setup drift monitoring

### Processing Jobs
- `POST /api/v1/processing/jobs` - Create job
- `GET /api/v1/processing/jobs/{id}` - Get job status
- `DELETE /api/v1/processing/jobs/{id}` - Cancel job

### Health
- `GET /health` - Basic health check
- `GET /api/v1/health` - Detailed health check
- `GET /api/v1/health/detailed` - System metrics
- `GET /api/v1/health/readiness` - Readiness check
- `GET /api/v1/health/liveness` - Liveness check

## Security Features

- JWT-based authentication
- Role-based access control
- Rate limiting (60 requests/minute per IP)
- Request validation and sanitization
- CORS protection
- Security headers (X-Frame-Options, X-XSS-Protection, etc.)

## Error Handling

The API returns structured error responses:

```json
{
  "error": {
    "code": 400,
    "message": "Invalid request data",
    "type": "ValidationError"
  }
}
```

## Rate Limiting

- Default: 60 requests per minute per IP address
- Rate limit headers included in responses
- 429 status code when limit exceeded

## Testing

```bash
# Run tests
pytest ai_data_readiness/api/tests/

# Run with coverage
pytest --cov=ai_data_readiness.api ai_data_readiness/api/tests/
```

## Configuration

Environment variables:

- `DB_HOST` - Database host (default: localhost)
- `DB_PORT` - Database port (default: 5432)
- `DB_NAME` - Database name (default: ai_data_readiness)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password
- `MAX_WORKERS` - Processing workers (default: 4)
- `BATCH_SIZE` - Processing batch size (default: 1000)
- `MEMORY_LIMIT_GB` - Memory limit (default: 8)

## Monitoring

- Health check endpoints for load balancers
- Prometheus metrics (if enabled)
- Structured logging
- Performance monitoring

## Production Deployment

```bash
# Using Docker
docker build -t ai-data-readiness-api .
docker run -p 8000:8000 ai-data-readiness-api

# Using systemd
sudo systemctl start ai-data-readiness-api
sudo systemctl enable ai-data-readiness-api
```