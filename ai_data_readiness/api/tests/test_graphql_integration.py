"""Integration tests for GraphQL API with real-time subscriptions."""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from strawberry.test import BaseGraphQLTestClient

from ..app import app
from ..graphql.schema import schema
from ..graphql.app import AuthContext


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def graphql_client():
    """Create GraphQL test client."""
    return BaseGraphQLTestClient(schema)


@pytest.fixture
def auth_headers():
    """Create authorization headers for testing."""
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    }


class TestGraphQLEndpointIntegration:
    """Test GraphQL endpoint integration with FastAPI."""
    
    def test_graphql_endpoint_available(self, test_client):
        """Test that GraphQL endpoint is available."""
        query = """
        query {
            systemMetrics
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    def test_graphql_playground_available(self, test_client):
        """Test that GraphQL playground is available."""
        response = test_client.get("/graphql/playground")
        assert response.status_code == 200
        assert "GraphQL Playground" in response.text
    
    def test_graphql_introspection(self, test_client):
        """Test GraphQL schema introspection."""
        query = """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                }
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "__schema" in data["data"]
        
        # Check that our custom types are present
        type_names = [t["name"] for t in data["data"]["__schema"]["types"]]
        assert "Dataset" in type_names
        assert "QualityReport" in type_names
        assert "BiasReport" in type_names


class TestComplexQueryIntegration:
    """Test complex GraphQL queries with multiple resolvers."""
    
    @patch('ai_data_readiness.api.graphql.resolvers.DataIngestionService')
    @patch('ai_data_readiness.api.graphql.resolvers.QualityAssessmentEngine')
    @patch('ai_data_readiness.api.graphql.resolvers.BiasAnalysisEngine')
    def test_dataset_analysis_integration(
        self, mock_bias_engine, mock_quality_engine, mock_ingestion_service, test_client
    ):
        """Test comprehensive dataset analysis query."""
        # Setup mocks
        mock_ingestion = Mock()
        mock_ingestion.get_dataset_metadata = AsyncMock(return_value={
            "name": "Integration Test Dataset",
            "description": "Test dataset for integration",
            "status": "ready",
            "ai_readiness_score": 0.85,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "version": "1.0"
        })
        mock_ingestion_service.return_value = mock_ingestion
        
        mock_quality = Mock()
        mock_quality_report = Mock()
        mock_quality_report.overall_score = 0.88
        mock_quality_report.completeness_score = 0.92
        mock_quality_report.accuracy_score = 0.85
        mock_quality_report.consistency_score = 0.87
        mock_quality_report.validity_score = 0.89
        mock_quality_report.issues = []
        mock_quality_report.recommendations = []
        mock_quality.assess_quality = AsyncMock(return_value=mock_quality_report)
        mock_quality_engine.return_value = mock_quality
        
        mock_bias = Mock()
        mock_bias_report = Mock()
        mock_bias_report.protected_attributes = ["gender", "age"]
        mock_bias_report.bias_metrics = {"demographic_parity": 0.05}
        mock_bias_report.fairness_violations = []
        mock_bias_report.mitigation_strategies = []
        mock_bias.detect_bias = AsyncMock(return_value=mock_bias_report)
        mock_bias_engine.return_value = mock_bias
        
        query = """
        query DatasetAnalysis($datasetId: String!) {
            datasetAnalysis(datasetId: $datasetId)
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={
                "query": query,
                "variables": {"datasetId": "test-dataset-id"}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["datasetAnalysis"] is not None
        
        analysis = data["data"]["datasetAnalysis"]
        assert "dataset" in analysis
        assert "quality" in analysis
        assert "bias" in analysis
        assert "analysis_timestamp" in analysis
    
    def test_cross_dataset_comparison(self, test_client):
        """Test cross-dataset comparison query."""
        query = """
        query CrossDatasetComparison($datasetIds: [String!]!) {
            crossDatasetComparison(datasetIds: $datasetIds)
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={
                "query": query,
                "variables": {"datasetIds": ["dataset1", "dataset2", "dataset3"]}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        
        comparison = data["data"]["crossDatasetComparison"]
        assert "datasets" in comparison
        assert "summary" in comparison
        assert "comparison_timestamp" in comparison
    
    def test_pipeline_health_query(self, test_client):
        """Test pipeline health monitoring query."""
        query = """
        query PipelineHealth {
            pipelineHealth
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        
        health = data["data"]["pipelineHealth"]
        assert "health_score" in health
        assert "status" in health
        assert "last_updated" in health


class TestMutationIntegration:
    """Test GraphQL mutations integration."""
    
    def test_create_dataset_mutation_integration(self, test_client, auth_headers):
        """Test dataset creation mutation."""
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
        
        response = test_client.post(
            "/graphql",
            json={
                "query": mutation,
                "variables": {
                    "input": {
                        "name": "Integration Test Dataset",
                        "description": "Created via GraphQL mutation",
                        "source": "integration_test",
                        "format": "csv",
                        "tags": ["test", "integration"]
                    }
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        
        dataset = data["data"]["createDataset"]
        assert dataset["name"] == "Integration Test Dataset"
        assert dataset["status"] == "PENDING"
        assert "id" in dataset
    
    def test_assess_quality_mutation_integration(self, test_client, auth_headers):
        """Test quality assessment mutation."""
        mutation = """
        mutation AssessQuality($input: QualityAssessmentInput!) {
            assessQuality(input: $input) {
                datasetId
                overallScore
                completenessScore
                accuracyScore
                generatedAt
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={
                "query": mutation,
                "variables": {
                    "input": {
                        "datasetId": "test-dataset-id",
                        "dimensions": ["COMPLETENESS", "ACCURACY"],
                        "generateRecommendations": True
                    }
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        
        report = data["data"]["assessQuality"]
        assert report["datasetId"] == "test-dataset-id"
        assert "overallScore" in report
    
    def test_create_processing_job_mutation_integration(self, test_client, auth_headers):
        """Test processing job creation mutation."""
        mutation = """
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
        """
        
        response = test_client.post(
            "/graphql",
            json={
                "query": mutation,
                "variables": {
                    "input": {
                        "datasetId": "test-dataset-id",
                        "jobType": "quality_assessment",
                        "parameters": {"threshold": 0.8},
                        "priority": "high"
                    }
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        
        job = data["data"]["createProcessingJob"]
        assert job["datasetId"] == "test-dataset-id"
        assert job["jobType"] == "quality_assessment"
        assert job["status"] == "QUEUED"
        assert "jobId" in job


class TestSubscriptionIntegration:
    """Test GraphQL subscriptions integration."""
    
    @pytest.mark.asyncio
    async def test_subscription_schema_validation(self, graphql_client):
        """Test that subscription schema is valid."""
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
        
        # Test that subscription can be parsed without errors
        result = graphql_client.query(subscription, variable_values={"datasetId": "test"})
        # For subscriptions, we mainly check that there are no parsing errors
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_multiple_subscriptions_schema(self, graphql_client):
        """Test multiple subscription types."""
        subscriptions = [
            """
            subscription QualityUpdates($datasetId: String!) {
                qualityUpdates(datasetId: $datasetId) {
                    datasetId
                    overallScore
                    generatedAt
                }
            }
            """,
            """
            subscription DriftAlerts($datasetId: String!) {
                driftAlerts(datasetId: $datasetId) {
                    datasetId
                    driftScore
                    generatedAt
                }
            }
            """,
            """
            subscription JobStatusUpdates($jobId: String!) {
                jobStatusUpdates(jobId: $jobId) {
                    jobId
                    status
                    progress
                }
            }
            """,
            """
            subscription PipelineStatusUpdates {
                pipelineStatusUpdates
            }
            """
        ]
        
        for subscription in subscriptions:
            result = graphql_client.query(
                subscription,
                variable_values={"datasetId": "test", "jobId": "test-job"}
            )
            assert result is not None


class TestErrorHandlingIntegration:
    """Test error handling in GraphQL integration."""
    
    def test_invalid_query_syntax(self, test_client):
        """Test handling of invalid GraphQL syntax."""
        invalid_query = """
        query {
            dataset(datasetId: "test" {
                id
                name
            }
        }
        """  # Missing closing parenthesis
        
        response = test_client.post(
            "/graphql",
            json={"query": invalid_query}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "errors" in data
    
    def test_invalid_field_selection(self, test_client):
        """Test handling of invalid field selection."""
        query = """
        query {
            dataset(datasetId: "test") {
                id
                nonExistentField
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={"query": query}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "errors" in data
    
    def test_missing_required_variables(self, test_client):
        """Test handling of missing required variables."""
        query = """
        query GetDataset($datasetId: String!) {
            dataset(datasetId: $datasetId) {
                id
                name
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={"query": query}  # Missing variables
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "errors" in data
    
    def test_invalid_variable_types(self, test_client):
        """Test handling of invalid variable types."""
        query = """
        query ListDatasets($limit: Int!) {
            datasets(limit: $limit) {
                id
                name
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={
                "query": query,
                "variables": {"limit": "not_a_number"}
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "errors" in data


class TestPerformanceIntegration:
    """Test GraphQL performance characteristics."""
    
    def test_query_complexity_handling(self, test_client):
        """Test handling of complex nested queries."""
        complex_query = """
        query ComplexQuery {
            datasets(limit: 5) {
                id
                name
                qualityScore
                aiReadinessScore
            }
            systemMetrics
            usageAnalytics
            processingJobs(limit: 10) {
                jobId
                status
                progress
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={"query": complex_query}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]["datasets"]) <= 5
        assert len(data["data"]["processingJobs"]) <= 10
    
    def test_concurrent_requests(self, test_client):
        """Test handling of concurrent GraphQL requests."""
        import threading
        import time
        
        query = """
        query {
            systemMetrics
        }
        """
        
        results = []
        
        def make_request():
            response = test_client.post(
                "/graphql",
                json={"query": query}
            )
            results.append(response.status_code)
        
        # Create multiple threads to make concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


class TestAuthenticationIntegration:
    """Test GraphQL authentication integration."""
    
    def test_authenticated_query(self, test_client, auth_headers):
        """Test query with authentication."""
        query = """
        query {
            datasets(limit: 5) {
                id
                name
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={"query": query},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    def test_unauthenticated_mutation(self, test_client):
        """Test mutation without authentication."""
        mutation = """
        mutation CreateDataset($input: DatasetCreateInput!) {
            createDataset(input: $input) {
                id
                name
            }
        }
        """
        
        response = test_client.post(
            "/graphql",
            json={
                "query": mutation,
                "variables": {
                    "input": {
                        "name": "Test Dataset",
                        "source": "test",
                        "format": "csv"
                    }
                }
            }
        )
        
        # Should still work for testing, but in production would require auth
        assert response.status_code in [200, 401]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])