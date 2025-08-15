# API Documentation

The AI Data Readiness Platform provides comprehensive REST and GraphQL APIs for programmatic access to all platform capabilities.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://ai-data-readiness.example.com`

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Getting a Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your-username", "password": "your-password"}'
```

## REST API Endpoints

### Health Check

#### GET /health
Check the health status of the platform.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

### Dataset Management

#### POST /api/v1/datasets/upload
Upload a new dataset for analysis.

**Parameters:**
- `file` (file): Dataset file (CSV, JSON, Parquet, etc.)
- `name` (string): Dataset name
- `description` (string, optional): Dataset description

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "name": "My Dataset",
  "status": "uploaded",
  "file_size": 1024000,
  "rows": 10000,
  "columns": 25
}
```

#### GET /api/v1/datasets
List all datasets.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20)
- `search` (string): Search term

**Response:**
```json
{
  "datasets": [
    {
      "id": "uuid-string",
      "name": "Dataset 1",
      "created_at": "2024-01-15T10:30:00Z",
      "quality_score": 0.85,
      "ai_readiness_score": 0.78
    }
  ],
  "total": 100,
  "page": 1,
  "limit": 20
}
```

#### GET /api/v1/datasets/{id}
Get dataset details.

**Response:**
```json
{
  "id": "uuid-string",
  "name": "My Dataset",
  "description": "Sample dataset",
  "schema": {
    "columns": [
      {
        "name": "age",
        "type": "integer",
        "nullable": false
      }
    ]
  },
  "metadata": {
    "rows": 10000,
    "columns": 25,
    "file_size": 1024000
  },
  "quality_score": 0.85,
  "ai_readiness_score": 0.78
}
```

### Quality Assessment

#### GET /api/v1/datasets/{id}/quality
Get comprehensive quality assessment for a dataset.

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "overall_score": 0.85,
  "completeness_score": 0.92,
  "accuracy_score": 0.88,
  "consistency_score": 0.81,
  "validity_score": 0.79,
  "issues": [
    {
      "type": "missing_values",
      "column": "age",
      "severity": "medium",
      "count": 150,
      "percentage": 1.5
    }
  ],
  "recommendations": [
    {
      "type": "imputation",
      "column": "age",
      "method": "median",
      "priority": "high"
    }
  ]
}
```

#### POST /api/v1/datasets/{id}/quality/assess
Trigger a new quality assessment.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "estimated_duration": 300
}
```

### Bias Analysis

#### GET /api/v1/datasets/{id}/bias
Get bias analysis results.

**Query Parameters:**
- `protected_attributes` (array): List of protected attributes to analyze

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "protected_attributes": ["gender", "race"],
  "bias_score": 0.23,
  "fairness_metrics": {
    "demographic_parity": 0.15,
    "equalized_odds": 0.18,
    "calibration": 0.12
  },
  "violations": [
    {
      "attribute": "gender",
      "metric": "demographic_parity",
      "value": 0.25,
      "threshold": 0.20,
      "severity": "high"
    }
  ],
  "mitigation_strategies": [
    {
      "strategy": "resampling",
      "attribute": "gender",
      "method": "smote",
      "expected_improvement": 0.15
    }
  ]
}
```

### Feature Engineering

#### GET /api/v1/datasets/{id}/features/recommendations
Get feature engineering recommendations.

**Query Parameters:**
- `model_type` (string): Target model type (classification, regression, etc.)
- `target_column` (string): Target variable name

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "model_type": "classification",
  "target_column": "outcome",
  "recommendations": [
    {
      "type": "encoding",
      "column": "category",
      "method": "one_hot",
      "priority": "high",
      "expected_improvement": 0.12
    },
    {
      "type": "scaling",
      "columns": ["age", "income"],
      "method": "standard",
      "priority": "medium",
      "expected_improvement": 0.08
    }
  ],
  "feature_importance": {
    "age": 0.25,
    "income": 0.18,
    "category": 0.15
  }
}
```

#### POST /api/v1/datasets/{id}/features/transform
Apply feature transformations.

**Request Body:**
```json
{
  "transformations": [
    {
      "type": "encoding",
      "column": "category",
      "method": "one_hot"
    },
    {
      "type": "scaling",
      "columns": ["age", "income"],
      "method": "standard"
    }
  ]
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "transformations_count": 2,
  "estimated_duration": 180
}
```

### Compliance Validation

#### GET /api/v1/datasets/{id}/compliance
Get compliance validation results.

**Query Parameters:**
- `regulations` (array): Regulations to check (gdpr, ccpa, etc.)

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "regulations": ["gdpr", "ccpa"],
  "compliance_score": 0.78,
  "violations": [
    {
      "regulation": "gdpr",
      "article": "Article 9",
      "description": "Special category data detected",
      "columns": ["health_status"],
      "severity": "high"
    }
  ],
  "recommendations": [
    {
      "type": "anonymization",
      "columns": ["health_status"],
      "method": "k_anonymity",
      "k_value": 5
    }
  ]
}
```

### Data Lineage

#### GET /api/v1/datasets/{id}/lineage
Get data lineage information.

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "lineage": {
    "source_datasets": [
      {
        "id": "source-uuid",
        "name": "Raw Data",
        "transformation": "initial_upload"
      }
    ],
    "transformations": [
      {
        "id": "transform-uuid",
        "type": "feature_engineering",
        "timestamp": "2024-01-15T10:30:00Z",
        "user": "user@example.com"
      }
    ],
    "derived_datasets": [
      {
        "id": "derived-uuid",
        "name": "Processed Data",
        "transformation": "bias_mitigation"
      }
    ]
  }
}
```

### Drift Monitoring

#### GET /api/v1/datasets/{id}/drift
Get drift monitoring results.

**Query Parameters:**
- `reference_dataset_id` (string): Reference dataset for comparison
- `time_window` (string): Time window for analysis (1d, 7d, 30d)

**Response:**
```json
{
  "dataset_id": "uuid-string",
  "reference_dataset_id": "reference-uuid",
  "drift_score": 0.15,
  "feature_drift_scores": {
    "age": 0.05,
    "income": 0.25,
    "category": 0.12
  },
  "statistical_tests": {
    "ks_test": {
      "statistic": 0.15,
      "p_value": 0.001,
      "significant": true
    }
  },
  "alerts": [
    {
      "feature": "income",
      "severity": "high",
      "drift_score": 0.25,
      "threshold": 0.20
    }
  ]
}
```

### Job Management

#### GET /api/v1/jobs/{id}
Get job status and results.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 100,
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z",
  "result": {
    "output_dataset_id": "output-uuid",
    "metrics": {
      "processing_time": 300,
      "rows_processed": 10000
    }
  }
}
```

## GraphQL API

The platform provides a GraphQL endpoint at `/graphql` for complex queries and real-time subscriptions.

### Example Queries

#### Get Dataset with Quality and Bias Information

```graphql
query GetDatasetDetails($id: ID!) {
  dataset(id: $id) {
    id
    name
    description
    qualityReport {
      overallScore
      completenessScore
      accuracyScore
      issues {
        type
        column
        severity
      }
    }
    biasReport {
      biasScore
      protectedAttributes
      violations {
        attribute
        metric
        severity
      }
    }
  }
}
```

#### List Datasets with Filtering

```graphql
query ListDatasets($filter: DatasetFilter, $pagination: Pagination) {
  datasets(filter: $filter, pagination: $pagination) {
    nodes {
      id
      name
      qualityScore
      aiReadinessScore
      createdAt
    }
    totalCount
    pageInfo {
      hasNextPage
      hasPreviousPage
    }
  }
}
```

### Subscriptions

#### Real-time Job Updates

```graphql
subscription JobUpdates($jobId: ID!) {
  jobUpdates(jobId: $jobId) {
    id
    status
    progress
    message
  }
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format",
    "details": {
      "field": "file",
      "allowed_formats": ["csv", "json", "parquet"]
    }
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **General API**: 100 requests per minute
- **Upload endpoints**: 10 requests per minute
- **Heavy processing**: 5 requests per minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## SDK and Client Libraries

Official SDKs are available for popular programming languages:

- [Python SDK](https://github.com/your-org/ai-data-readiness-python)
- [JavaScript SDK](https://github.com/your-org/ai-data-readiness-js)
- [R Package](https://github.com/your-org/ai-data-readiness-r)

### Python SDK Example

```python
from ai_data_readiness import Client

client = Client(api_key="your-api-key")

# Upload dataset
dataset = client.datasets.upload("data.csv", name="My Dataset")

# Get quality assessment
quality = client.datasets.get_quality(dataset.id)
print(f"Quality Score: {quality.overall_score}")

# Check for bias
bias = client.datasets.get_bias(dataset.id, protected_attributes=["gender"])
print(f"Bias Score: {bias.bias_score}")
```