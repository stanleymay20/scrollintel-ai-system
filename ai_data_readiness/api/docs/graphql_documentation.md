# GraphQL API Documentation

## Overview

The AI Data Readiness Platform provides a comprehensive GraphQL API for flexible data querying, complex dataset relationships, and real-time updates through subscriptions. This API enables developers to efficiently retrieve and manipulate data across multiple dimensions including quality assessment, bias analysis, feature engineering, compliance checking, and data lineage.

## Features

- **Flexible Querying**: Query exactly the data you need with GraphQL's powerful selection syntax
- **Complex Relationships**: Navigate between datasets, quality reports, bias analyses, and lineage information
- **Real-time Updates**: Subscribe to live updates for datasets, processing jobs, and system alerts
- **Type Safety**: Strongly typed schema with comprehensive validation
- **Authentication**: Integrated with the platform's authentication system
- **Performance**: Optimized resolvers with efficient data loading

## Endpoints

- **GraphQL Endpoint**: `/graphql`
- **GraphQL Playground**: `/graphql/playground` (development only)
- **Schema Introspection**: Available via standard GraphQL introspection queries

## Schema Overview

### Core Types

#### Dataset
```graphql
type Dataset {
  id: String!
  name: String!
  description: String!
  schema: Schema
  metadata: DatasetMetadata
  qualityScore: Float!
  aiReadinessScore: Float!
  status: DatasetStatus!
  createdAt: DateTime!
  updatedAt: DateTime!
  version: String!
  
  # Computed fields
  isAiReady(threshold: Float = 0.8): Boolean!
  qualityGrade: String!
}
```

#### QualityReport
```graphql
type QualityReport {
  datasetId: String!
  overallScore: Float!
  completenessScore: Float!
  accuracyScore: Float!
  consistencyScore: Float!
  validityScore: Float!
  uniquenessScore: Float!
  timelinessScore: Float!
  issues: [QualityIssue!]!
  recommendations: [Recommendation!]!
  generatedAt: DateTime!
  
  # Computed fields
  dimensionScores: JSON!
}
```

#### BiasReport
```graphql
type BiasReport {
  datasetId: String!
  protectedAttributes: [String!]!
  biasMetrics: JSON!
  fairnessViolations: [FairnessViolation!]!
  mitigationStrategies: [MitigationStrategy!]!
  generatedAt: DateTime!
}
```

#### ProcessingJob
```graphql
type ProcessingJob {
  jobId: String!
  datasetId: String!
  jobType: String!
  status: JobStatus!
  progress: Float!
  parameters: JSON!
  result: JSON
  errorMessage: String
  createdAt: DateTime!
  startedAt: DateTime
  completedAt: DateTime
  
  # Computed fields
  durationSeconds: Int
  isRunning: Boolean!
}
```

### Enums

```graphql
enum DatasetStatus {
  PENDING
  PROCESSING
  READY
  ERROR
  ARCHIVED
}

enum JobStatus {
  QUEUED
  RUNNING
  COMPLETED
  FAILED
  CANCELLED
}

enum QualityDimension {
  COMPLETENESS
  ACCURACY
  CONSISTENCY
  VALIDITY
  UNIQUENESS
  TIMELINESS
}
```

## Queries

### Basic Dataset Queries

#### Get Single Dataset
```graphql
query GetDataset($datasetId: String!) {
  dataset(datasetId: $datasetId) {
    id
    name
    description
    qualityScore
    aiReadinessScore
    status
    isAiReady(threshold: 0.8)
    qualityGrade
    createdAt
    updatedAt
  }
}
```

#### List Datasets with Filtering
```graphql
query ListDatasets($limit: Int, $offset: Int, $status: DatasetStatus, $minQualityScore: Float) {
  datasets(limit: $limit, offset: $offset, status: $status, minQualityScore: $minQualityScore) {
    id
    name
    description
    qualityScore
    aiReadinessScore
    status
    createdAt
  }
}
```

#### Search Datasets
```graphql
query SearchDatasets($query: String!, $tags: [String!], $limit: Int) {
  searchDatasets(query: $query, tags: $tags, limit: $limit) {
    id
    name
    description
    qualityScore
    tags
  }
}
```

### Quality Assessment Queries

#### Get Quality Report
```graphql
query GetQualityReport($datasetId: String!) {
  qualityReport(datasetId: $datasetId) {
    datasetId
    overallScore
    completenessScore
    accuracyScore
    consistencyScore
    validityScore
    dimensionScores
    issues {
      dimension
      severity
      description
      affectedColumns
      affectedRows
      recommendation
    }
    recommendations {
      type
      priority
      title
      description
      actionItems
      estimatedImpact
    }
    generatedAt
  }
}
```

#### Get AI Readiness Assessment
```graphql
query GetAIReadiness($datasetId: String!) {
  aiReadiness(datasetId: $datasetId) {
    overallScore
    dataQualityScore
    featureQualityScore
    biasScore
    complianceScore
    scalabilityScore
    dimensions
    improvementAreas {
      area
      currentScore
      targetScore
      priority
      actions
    }
    generatedAt
  }
}
```

#### Get Quality Trends
```graphql
query GetQualityTrends($datasetId: String!, $days: Int = 30) {
  qualityTrends(datasetId: $datasetId, days: $days) {
    datasetId
    overallScore
    completenessScore
    accuracyScore
    generatedAt
  }
}
```

### Bias Analysis Queries

#### Get Bias Report
```graphql
query GetBiasReport($datasetId: String!) {
  biasReport(datasetId: $datasetId) {
    datasetId
    protectedAttributes
    biasMetrics
    fairnessViolations {
      biasType
      protectedAttribute
      severity
      description
      metricValue
      threshold
      affectedGroups
    }
    mitigationStrategies {
      strategyType
      description
      implementationSteps
      expectedImpact
      complexity
    }
    generatedAt
  }
}
```

#### Compare Bias Across Datasets
```graphql
query CompareBias($datasetIds: [String!]!) {
  biasComparison(datasetIds: $datasetIds) {
    datasetId
    protectedAttributes
    biasMetrics
    generatedAt
  }
}
```

### Feature Engineering Queries

#### Get Feature Recommendations
```graphql
query GetFeatureRecommendations($datasetId: String!) {
  featureRecommendations(datasetId: $datasetId) {
    datasetId
    modelType
    recommendations {
      featureName
      transformation
      description
      priority
      expectedImpact
    }
    transformations
    encodingStrategies
    generatedAt
  }
}
```

#### Analyze Feature Impact
```graphql
query AnalyzeFeatureImpact($datasetId: String!, $features: [String!]!) {
  featureImpactAnalysis(datasetId: $datasetId, features: $features)
}
```

### Data Lineage Queries

#### Get Data Lineage
```graphql
query GetLineage($datasetId: String!) {
  lineage(datasetId: $datasetId) {
    datasetId
    sourceDatasets
    transformations {
      type
      description
      parameters
      timestamp
    }
    downstreamDatasets
    modelsTrained
    createdBy
    createdAt
  }
}
```

#### Get Lineage Graph
```graphql
query GetLineageGraph($datasetId: String!) {
  lineageGraph(datasetId: $datasetId)
}
```

#### Analyze Impact
```graphql
query AnalyzeImpact($datasetId: String!) {
  impactAnalysis(datasetId: $datasetId)
}
```

### Processing Job Queries

#### Get Processing Job
```graphql
query GetProcessingJob($jobId: String!) {
  processingJob(jobId: $jobId) {
    jobId
    datasetId
    jobType
    status
    progress
    parameters
    result
    errorMessage
    createdAt
    startedAt
    completedAt
    durationSeconds
    isRunning
  }
}
```

#### List Processing Jobs
```graphql
query ListProcessingJobs($datasetId: String, $status: JobStatus, $limit: Int) {
  processingJobs(datasetId: $datasetId, status: $status, limit: $limit) {
    jobId
    datasetId
    jobType
    status
    progress
    createdAt
  }
}
```

### System Queries

#### Get System Metrics
```graphql
query GetSystemMetrics {
  systemMetrics
}
```

#### Get Usage Analytics
```graphql
query GetUsageAnalytics {
  usageAnalytics
}
```

### Complex Relationship Queries

#### Comprehensive Dataset Analysis
```graphql
query DatasetAnalysis($datasetId: String!) {
  datasetAnalysis(datasetId: $datasetId)
}
```

#### Cross-Dataset Comparison
```graphql
query CrossDatasetComparison($datasetIds: [String!]!) {
  crossDatasetComparison(datasetIds: $datasetIds)
}
```

#### Pipeline Health
```graphql
query PipelineHealth {
  pipelineHealth
}
```

## Mutations

### Dataset Mutations

#### Create Dataset
```graphql
mutation CreateDataset($input: DatasetCreateInput!) {
  createDataset(input: $input) {
    id
    name
    description
    status
    createdAt
  }
}
```

#### Update Dataset
```graphql
mutation UpdateDataset($datasetId: String!, $input: DatasetUpdateInput!) {
  updateDataset(datasetId: $datasetId, input: $input) {
    id
    name
    description
    updatedAt
  }
}
```

#### Delete Dataset
```graphql
mutation DeleteDataset($datasetId: String!) {
  deleteDataset(datasetId: $datasetId)
}
```

### Quality Assessment Mutations

#### Assess Quality
```graphql
mutation AssessQuality($input: QualityAssessmentInput!) {
  assessQuality(input: $input) {
    datasetId
    overallScore
    completenessScore
    accuracyScore
    consistencyScore
    validityScore
    generatedAt
  }
}
```

### Bias Analysis Mutations

#### Analyze Bias
```graphql
mutation AnalyzeBias($input: BiasAnalysisInput!) {
  analyzeBias(input: $input) {
    datasetId
    protectedAttributes
    biasMetrics
    fairnessViolations {
      biasType
      severity
      description
    }
    generatedAt
  }
}
```

### Feature Engineering Mutations

#### Generate Feature Recommendations
```graphql
mutation GenerateFeatures($input: FeatureEngineeringInput!) {
  generateFeatures(input: $input) {
    datasetId
    modelType
    recommendations {
      featureName
      transformation
      priority
      expectedImpact
    }
    generatedAt
  }
}
```

### Processing Job Mutations

#### Create Processing Job
```graphql
mutation CreateProcessingJob($input: ProcessingJobInput!) {
  createProcessingJob(input: $input) {
    jobId
    datasetId
    jobType
    status
    progress
    createdAt
  }
}
```

#### Cancel Processing Job
```graphql
mutation CancelProcessingJob($jobId: String!) {
  cancelProcessingJob(jobId: $jobId)
}
```

## Subscriptions

### Dataset Subscriptions

#### Dataset Updates
```graphql
subscription DatasetUpdates($datasetId: String!) {
  datasetUpdates(datasetId: $datasetId) {
    id
    name
    qualityScore
    aiReadinessScore
    status
    updatedAt
  }
}
```

### Quality Subscriptions

#### Quality Updates
```graphql
subscription QualityUpdates($datasetId: String!) {
  qualityUpdates(datasetId: $datasetId) {
    datasetId
    overallScore
    completenessScore
    accuracyScore
    generatedAt
  }
}
```

### Processing Subscriptions

#### Job Status Updates
```graphql
subscription JobStatusUpdates($jobId: String!) {
  jobStatusUpdates(jobId: $jobId) {
    jobId
    status
    progress
    errorMessage
    updatedAt
  }
}
```

#### Pipeline Status Updates
```graphql
subscription PipelineStatusUpdates {
  pipelineStatusUpdates
}
```

### Drift Monitoring Subscriptions

#### Drift Alerts
```graphql
subscription DriftAlerts($datasetId: String!) {
  driftAlerts(datasetId: $datasetId) {
    datasetId
    driftScore
    alerts {
      feature
      driftScore
      severity
      timestamp
    }
    generatedAt
  }
}
```

### System Subscriptions

#### System Alerts
```graphql
subscription SystemAlerts {
  systemAlerts
}
```

#### Compliance Alerts
```graphql
subscription ComplianceAlerts {
  complianceAlerts {
    datasetId
    complianceScore
    violations {
      regulation
      violationType
      severity
    }
    generatedAt
  }
}
```

## Input Types

### DatasetCreateInput
```graphql
input DatasetCreateInput {
  name: String!
  description: String = ""
  source: String!
  format: String!
  tags: [String!] = []
}
```

### QualityAssessmentInput
```graphql
input QualityAssessmentInput {
  datasetId: String!
  dimensions: [QualityDimension!] = []
  generateRecommendations: Boolean = true
}
```

### BiasAnalysisInput
```graphql
input BiasAnalysisInput {
  datasetId: String!
  protectedAttributes: [String!]!
  targetColumn: String
  biasTypes: [String!] = []
}
```

### ProcessingJobInput
```graphql
input ProcessingJobInput {
  datasetId: String!
  jobType: String!
  parameters: JSON = {}
  priority: String = "normal"
}
```

## Error Handling

The GraphQL API uses standard GraphQL error handling:

```json
{
  "data": null,
  "errors": [
    {
      "message": "Dataset not found",
      "locations": [{"line": 2, "column": 3}],
      "path": ["dataset"],
      "extensions": {
        "code": "NOT_FOUND",
        "dataset_id": "invalid-id"
      }
    }
  ]
}
```

Common error codes:
- `NOT_FOUND`: Resource not found
- `VALIDATION_ERROR`: Input validation failed
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions
- `INTERNAL_ERROR`: Server error

## Authentication

Include authentication token in the Authorization header:

```
Authorization: Bearer <your-token>
```

## Rate Limiting

The API implements rate limiting:
- 1000 requests per hour for authenticated users
- 100 requests per hour for unauthenticated users
- Subscription connections are limited to 10 per user

## Best Practices

### Query Optimization
1. **Request only needed fields**: Use GraphQL's field selection to minimize data transfer
2. **Use fragments**: Reuse common field selections with fragments
3. **Batch related queries**: Combine related queries in a single request
4. **Implement pagination**: Use limit/offset for large result sets

### Subscription Management
1. **Close unused subscriptions**: Properly close WebSocket connections
2. **Handle reconnection**: Implement reconnection logic for network issues
3. **Filter server-side**: Use subscription variables to filter events

### Error Handling
1. **Check for errors**: Always check the errors field in responses
2. **Handle partial data**: GraphQL can return partial data with errors
3. **Implement retry logic**: Retry failed requests with exponential backoff

## Examples

### Complete Dataset Analysis
```graphql
query CompleteDatasetAnalysis($datasetId: String!) {
  dataset(datasetId: $datasetId) {
    id
    name
    description
    qualityScore
    aiReadinessScore
    status
    isAiReady(threshold: 0.8)
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
      priority
      title
      description
    }
  }
  
  biasReport(datasetId: $datasetId) {
    protectedAttributes
    biasMetrics
    fairnessViolations {
      biasType
      severity
    }
  }
  
  lineage(datasetId: $datasetId) {
    sourceDatasets
    downstreamDatasets
    modelsTrained
  }
}
```

### Real-time Monitoring Setup
```graphql
# Subscribe to multiple update streams
subscription MonitorDataset($datasetId: String!) {
  datasetUpdates(datasetId: $datasetId) {
    id
    qualityScore
    status
    updatedAt
  }
}

subscription MonitorQuality($datasetId: String!) {
  qualityUpdates(datasetId: $datasetId) {
    overallScore
    generatedAt
  }
}

subscription MonitorDrift($datasetId: String!) {
  driftAlerts(datasetId: $datasetId) {
    driftScore
    alerts {
      feature
      severity
    }
  }
}
```

This GraphQL API provides a powerful and flexible interface for interacting with the AI Data Readiness Platform, enabling complex queries, real-time updates, and efficient data access patterns.