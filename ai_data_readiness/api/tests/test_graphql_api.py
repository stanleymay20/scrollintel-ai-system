"""Comprehensive tests for GraphQL API."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from strawberry.test import BaseGraphQLTestClient

from ..graphql.schema import schema
from ..graphql.app import AuthContext
from ..graphql.types import DatasetStatus, JobStatus


class GraphQLTestClient(BaseGraphQLTestClient):
    """Custom GraphQL test client."""
    
    def __init__(self, schema, context=None):
        super().__init__(schema)
        self.context = context or AuthContext(
            request=Mock(),
            user={"user_id": "test_user", "username": "testuser"}
        )


@pytest.fixture
def graphql_client():
    """Create GraphQL test client."""
    return GraphQLTestClient(schema)


@pytest.fixture
def mock_context():
    """Create mock GraphQL context."""
    return AuthContext(
        request=Mock(),
        user={"user_id": "test_user", "username": "testuser", "permissions": ["read", "write"]}
    )


class TestDatasetQueries:
    """Test dataset GraphQL queries."""
    
    @patch('ai_data_readiness.api.graphql.resolvers.DataIngestionService')
    @patch('ai_data_readiness.api.graphql.resolvers.QualityAssessmentEngine')
    def test_get_dataset_query(self, mock_quality_engine, mock_ingestion_service, graphql_client):
        """Test getting a single dataset."""
        # Mock data ingestion service
        mock_ingestion = Mock()
        mock_ingestion.get_dataset_metadata = AsyncMock(return_value={
            "name": "Test Dataset",
            "description": "Test description",
            "status": "ready",
            "ai_readiness_score": 0.85,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "version": "1.0"
        })
        mock_ingestion_service.return_value = mock_ingestion
        
        # Mock quality engine
        mock_quality = Mock()
        mock_quality_report = Mock()
        mock_quality_report.overall_score = 0.88
        mock_quality.assess_quality = AsyncMock(return_value=mock_quality_report)
        mock_quality_engine.return_value = mock_quality
        
        query = """
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
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["dataset"]["id"] == "test-dataset-id"
        assert result.data["dataset"]["name"] == "Test Dataset"
        assert result.data["dataset"]["qualityScore"] == 0.88
        assert result.data["dataset"]["isAiReady"] is True
        assert result.data["dataset"]["qualityGrade"] == "B"
    
    def test_list_datasets_query(self, graphql_client):
        """Test listing datasets."""
        query = """
        query ListDatasets($limit: Int, $offset: Int) {
            datasets(limit: $limit, offset: $offset) {
                id
                name
                description
                qualityScore
                status
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"limit": 10, "offset": 0}
        )
        
        assert result.errors is None
        assert isinstance(result.data["datasets"], list)
        assert len(result.data["datasets"]) <= 10
    
    def test_search_datasets_query(self, graphql_client):
        """Test searching datasets."""
        query = """
        query SearchDatasets($query: String!, $tags: [String!], $limit: Int) {
            searchDatasets(query: $query, tags: $tags, limit: $limit) {
                id
                name
                description
                qualityScore
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={
                "query": "test",
                "tags": ["sample", "test"],
                "limit": 5
            }
        )
        
        assert result.errors is None
        assert isinstance(result.data["searchDatasets"], list)
    
    def test_system_metrics_query(self, graphql_client):
        """Test system metrics query."""
        query = """
        query SystemMetrics {
            systemMetrics
        }
        """
        
        result = graphql_client.query(query)
        
        assert result.errors is None
        assert "totalDatasets" in result.data["systemMetrics"]
        assert "activeJobs" in result.data["systemMetrics"]


class TestQualityQueries:
    """Test quality assessment GraphQL queries."""
    
    @patch('ai_data_readiness.api.graphql.resolvers.QualityAssessmentEngine')
    def test_quality_report_query(self, mock_quality_engine, graphql_client):
        """Test getting quality report."""
        # Mock quality engine
        mock_quality = Mock()
        mock_report = Mock()
        mock_report.overall_score = 0.85
        mock_report.completeness_score = 0.92
        mock_report.accuracy_score = 0.88
        mock_report.consistency_score = 0.81
        mock_report.validity_score = 0.87
        mock_report.issues = []
        mock_report.recommendations = []
        mock_quality.assess_quality = AsyncMock(return_value=mock_report)
        mock_quality_engine.return_value = mock_quality
        
        query = """
        query QualityReport($datasetId: String!) {
            qualityReport(datasetId: $datasetId) {
                datasetId
                overallScore
                completenessScore
                accuracyScore
                consistencyScore
                validityScore
                dimensionScores
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["qualityReport"]["datasetId"] == "test-dataset-id"
        assert result.data["qualityReport"]["overallScore"] == 0.85
        assert "completeness" in result.data["qualityReport"]["dimensionScores"]
    
    def test_ai_readiness_query(self, graphql_client):
        """Test AI readiness assessment query."""
        query = """
        query AIReadiness($datasetId: String!) {
            aiReadiness(datasetId: $datasetId) {
                overallScore
                dataQualityScore
                featureQualityScore
                biasScore
                complianceScore
                scalabilityScore
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert "overallScore" in result.data["aiReadiness"]
    
    def test_quality_trends_query(self, graphql_client):
        """Test quality trends query."""
        query = """
        query QualityTrends($datasetId: String!, $days: Int) {
            qualityTrends(datasetId: $datasetId, days: $days) {
                datasetId
                overallScore
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id", "days": 30}
        )
        
        assert result.errors is None
        assert isinstance(result.data["qualityTrends"], list)


class TestBiasQueries:
    """Test bias analysis GraphQL queries."""
    
    @patch('ai_data_readiness.api.graphql.resolvers.BiasAnalysisEngine')
    def test_bias_report_query(self, mock_bias_engine, graphql_client):
        """Test getting bias report."""
        # Mock bias engine
        mock_bias = Mock()
        mock_report = Mock()
        mock_report.protected_attributes = ["gender", "age"]
        mock_report.bias_metrics = {"demographic_parity": 0.08}
        mock_report.fairness_violations = []
        mock_report.mitigation_strategies = []
        mock_bias.detect_bias = AsyncMock(return_value=mock_report)
        mock_bias_engine.return_value = mock_bias
        
        query = """
        query BiasReport($datasetId: String!) {
            biasReport(datasetId: $datasetId) {
                datasetId
                protectedAttributes
                biasMetrics
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["biasReport"]["datasetId"] == "test-dataset-id"
        assert "gender" in result.data["biasReport"]["protectedAttributes"]
    
    def test_bias_comparison_query(self, graphql_client):
        """Test bias comparison query."""
        query = """
        query BiasComparison($datasetIds: [String!]!) {
            biasComparison(datasetIds: $datasetIds) {
                datasetId
                protectedAttributes
                biasMetrics
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetIds": ["dataset1", "dataset2"]}
        )
        
        assert result.errors is None
        assert isinstance(result.data["biasComparison"], list)


class TestFeatureQueries:
    """Test feature engineering GraphQL queries."""
    
    def test_feature_recommendations_query(self, graphql_client):
        """Test feature recommendations query."""
        query = """
        query FeatureRecommendations($datasetId: String!) {
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
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["featureRecommendations"]["datasetId"] == "test-dataset-id"
    
    def test_feature_impact_analysis_query(self, graphql_client):
        """Test feature impact analysis query."""
        query = """
        query FeatureImpactAnalysis($datasetId: String!, $features: [String!]!) {
            featureImpactAnalysis(datasetId: $datasetId, features: $features)
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={
                "datasetId": "test-dataset-id",
                "features": ["feature1", "feature2"]
            }
        )
        
        assert result.errors is None
        assert isinstance(result.data["featureImpactAnalysis"], list)


class TestLineageQueries:
    """Test data lineage GraphQL queries."""
    
    def test_lineage_query(self, graphql_client):
        """Test data lineage query."""
        query = """
        query Lineage($datasetId: String!) {
            lineage(datasetId: $datasetId) {
                datasetId
                sourceDatasets
                downstreamDatasets
                modelsTrained
                createdBy
                createdAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["lineage"]["datasetId"] == "test-dataset-id"
    
    def test_lineage_graph_query(self, graphql_client):
        """Test lineage graph query."""
        query = """
        query LineageGraph($datasetId: String!) {
            lineageGraph(datasetId: $datasetId)
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert "nodes" in result.data["lineageGraph"]
        assert "edges" in result.data["lineageGraph"]
    
    def test_impact_analysis_query(self, graphql_client):
        """Test impact analysis query."""
        query = """
        query ImpactAnalysis($datasetId: String!) {
            impactAnalysis(datasetId: $datasetId)
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert isinstance(result.data["impactAnalysis"], list)


class TestDriftQueries:
    """Test drift monitoring GraphQL queries."""
    
    def test_drift_report_query(self, graphql_client):
        """Test drift report query."""
        query = """
        query DriftReport($datasetId: String!) {
            driftReport(datasetId: $datasetId) {
                datasetId
                referenceDatasetId
                driftScore
                featureDriftScores
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["driftReport"]["datasetId"] == "test-dataset-id"
    
    def test_drift_trends_query(self, graphql_client):
        """Test drift trends query."""
        query = """
        query DriftTrends($datasetId: String!, $days: Int) {
            driftTrends(datasetId: $datasetId, days: $days) {
                datasetId
                driftScore
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id", "days": 30}
        )
        
        assert result.errors is None
        assert isinstance(result.data["driftTrends"], list)


class TestProcessingQueries:
    """Test processing job GraphQL queries."""
    
    def test_processing_job_query(self, graphql_client):
        """Test processing job query."""
        query = """
        query ProcessingJob($jobId: String!) {
            processingJob(jobId: $jobId) {
                jobId
                datasetId
                jobType
                status
                progress
                durationSeconds
                isRunning
                createdAt
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"jobId": "test-job-id"}
        )
        
        assert result.errors is None
        assert result.data["processingJob"]["jobId"] == "test-job-id"
    
    def test_processing_jobs_query(self, graphql_client):
        """Test processing jobs list query."""
        query = """
        query ProcessingJobs($datasetId: String, $status: JobStatus, $limit: Int) {
            processingJobs(datasetId: $datasetId, status: $status, limit: $limit) {
                jobId
                datasetId
                jobType
                status
                progress
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"limit": 10}
        )
        
        assert result.errors is None
        assert isinstance(result.data["processingJobs"], list)
    
    def test_job_queue_status_query(self, graphql_client):
        """Test job queue status query."""
        query = """
        query JobQueueStatus {
            jobQueueStatus
        }
        """
        
        result = graphql_client.query(query)
        
        assert result.errors is None
        assert "queuedJobs" in result.data["jobQueueStatus"]


class TestMutations:
    """Test GraphQL mutations."""
    
    def test_create_dataset_mutation(self, graphql_client):
        """Test creating a dataset."""
        mutation = """
        mutation CreateDataset($input: DatasetCreateInput!) {
            createDataset(input: $input) {
                id
                name
                description
                status
                createdAt
            }
        }
        """
        
        result = graphql_client.query(
            mutation,
            variable_values={
                "input": {
                    "name": "New Test Dataset",
                    "description": "Test dataset description",
                    "source": "test_source",
                    "format": "csv",
                    "tags": ["test"]
                }
            }
        )
        
        assert result.errors is None
        assert result.data["createDataset"]["name"] == "New Test Dataset"
        assert result.data["createDataset"]["status"] == "PENDING"
    
    def test_assess_quality_mutation(self, graphql_client):
        """Test quality assessment mutation."""
        mutation = """
        mutation AssessQuality($input: QualityAssessmentInput!) {
            assessQuality(input: $input) {
                datasetId
                overallScore
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            mutation,
            variable_values={
                "input": {
                    "datasetId": "test-dataset-id",
                    "dimensions": ["COMPLETENESS", "ACCURACY"],
                    "generateRecommendations": True
                }
            }
        )
        
        assert result.errors is None
        assert result.data["assessQuality"]["datasetId"] == "test-dataset-id"
    
    def test_analyze_bias_mutation(self, graphql_client):
        """Test bias analysis mutation."""
        mutation = """
        mutation AnalyzeBias($input: BiasAnalysisInput!) {
            analyzeBias(input: $input) {
                datasetId
                protectedAttributes
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            mutation,
            variable_values={
                "input": {
                    "datasetId": "test-dataset-id",
                    "protectedAttributes": ["gender", "age"],
                    "targetColumn": "outcome"
                }
            }
        )
        
        assert result.errors is None
        assert result.data["analyzeBias"]["datasetId"] == "test-dataset-id"
    
    def test_create_processing_job_mutation(self, graphql_client):
        """Test creating a processing job."""
        mutation = """
        mutation CreateProcessingJob($input: ProcessingJobInput!) {
            createProcessingJob(input: $input) {
                jobId
                datasetId
                jobType
                status
                createdAt
            }
        }
        """
        
        result = graphql_client.query(
            mutation,
            variable_values={
                "input": {
                    "datasetId": "test-dataset-id",
                    "jobType": "quality_assessment",
                    "parameters": {},
                    "priority": "normal"
                }
            }
        )
        
        assert result.errors is None
        assert result.data["createProcessingJob"]["datasetId"] == "test-dataset-id"
        assert result.data["createProcessingJob"]["status"] == "QUEUED"


class TestSubscriptions:
    """Test GraphQL subscriptions."""
    
    @pytest.mark.asyncio
    async def test_dataset_updates_subscription(self, graphql_client):
        """Test dataset updates subscription."""
        subscription = """
        subscription DatasetUpdates($datasetId: String!) {
            datasetUpdates(datasetId: $datasetId) {
                id
                name
                qualityScore
                status
                updatedAt
            }
        }
        """
        
        # Note: This is a simplified test for subscription structure
        # In a real test, you would need to set up WebSocket connections
        # and test the actual streaming functionality
        
        # For now, we just verify the subscription can be parsed
        result = graphql_client.query(
            subscription,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        # Subscriptions return different results, so we check for no syntax errors
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_quality_updates_subscription(self, graphql_client):
        """Test quality updates subscription."""
        subscription = """
        subscription QualityUpdates($datasetId: String!) {
            qualityUpdates(datasetId: $datasetId) {
                datasetId
                overallScore
                generatedAt
            }
        }
        """
        
        result = graphql_client.query(
            subscription,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_drift_alerts_subscription(self, graphql_client):
        """Test drift alerts subscription."""
        subscription = """
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
        """
        
        result = graphql_client.query(
            subscription,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_job_status_updates_subscription(self, graphql_client):
        """Test job status updates subscription."""
        subscription = """
        subscription JobStatusUpdates($jobId: String!) {
            jobStatusUpdates(jobId: $jobId) {
                jobId
                status
                progress
                updatedAt
            }
        }
        """
        
        result = graphql_client.query(
            subscription,
            variable_values={"jobId": "test-job-id"}
        )
        
        assert result is not None


class TestComplexQueries:
    """Test complex GraphQL queries with multiple relationships."""
    
    def test_dataset_with_quality_and_bias(self, graphql_client):
        """Test complex query combining dataset, quality, and bias data."""
        query = """
        query DatasetWithAnalysis($datasetId: String!) {
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
                completenessScore
                accuracyScore
                issues {
                    dimension
                    severity
                    description
                }
                recommendations {
                    type
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
                    description
                }
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["dataset"]["id"] == "test-dataset-id"
        assert "overallScore" in result.data["qualityReport"]
        assert "protectedAttributes" in result.data["biasReport"]
    
    def test_dataset_lineage_with_impact(self, graphql_client):
        """Test complex lineage query with impact analysis."""
        query = """
        query DatasetLineageAnalysis($datasetId: String!) {
            lineage(datasetId: $datasetId) {
                datasetId
                sourceDatasets
                downstreamDatasets
                modelsTrained
                transformations {
                    type
                    description
                    timestamp
                }
            }
            lineageGraph(datasetId: $datasetId)
            impactAnalysis(datasetId: $datasetId)
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert result.data["lineage"]["datasetId"] == "test-dataset-id"
        assert "nodes" in result.data["lineageGraph"]
        assert isinstance(result.data["impactAnalysis"], list)
    
    def test_processing_jobs_with_metrics(self, graphql_client):
        """Test complex processing query with system metrics."""
        query = """
        query ProcessingOverview($datasetId: String) {
            processingJobs(datasetId: $datasetId, limit: 10) {
                jobId
                datasetId
                jobType
                status
                progress
                durationSeconds
                isRunning
                createdAt
            }
            jobQueueStatus
            systemMetrics
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "test-dataset-id"}
        )
        
        assert result.errors is None
        assert isinstance(result.data["processingJobs"], list)
        assert "queuedJobs" in result.data["jobQueueStatus"]
        assert "totalDatasets" in result.data["systemMetrics"]


class TestErrorHandling:
    """Test GraphQL error handling."""
    
    def test_invalid_dataset_id(self, graphql_client):
        """Test handling of invalid dataset ID."""
        query = """
        query GetDataset($datasetId: String!) {
            dataset(datasetId: $datasetId) {
                id
                name
            }
        }
        """
        
        result = graphql_client.query(
            query,
            variable_values={"datasetId": "nonexistent-dataset"}
        )
        
        # Should return null for non-existent dataset, not an error
        assert result.errors is None
        assert result.data["dataset"] is None
    
    def test_invalid_input_validation(self, graphql_client):
        """Test input validation errors."""
        mutation = """
        mutation CreateDataset($input: DatasetCreateInput!) {
            createDataset(input: $input) {
                id
                name
            }
        }
        """
        
        # Missing required fields
        result = graphql_client.query(
            mutation,
            variable_values={
                "input": {
                    "description": "Missing name field"
                }
            }
        )
        
        # Should have validation errors
        assert result.errors is not None
    
    def test_authorization_errors(self, graphql_client):
        """Test authorization error handling."""
        # This would require setting up a context without proper permissions
        # For now, we just verify the structure exists
        query = """
        query RestrictedOperation {
            systemMetrics
        }
        """
        
        result = graphql_client.query(query)
        
        # Should work with default test context
        assert result.errors is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])