"""
End-to-End Workflow Tests
Tests complete user workflows from data upload to insights
"""
import pytest
import asyncio
import json
import io
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi import UploadFile

from scrollintel.api.gateway import app
from scrollintel.engines.file_processor import FileProcessor
from scrollintel.engines.automodel_engine import AutoModelEngine
from scrollintel.engines.scroll_viz_engine import ScrollVizEngine
from scrollintel.engines.scroll_qa_engine import ScrollQAEngine


class TestEndToEndWorkflows:
    """Test complete user workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_data_analysis_workflow(self, test_client, test_user_token, temp_files, mock_ai_services):
        """Test complete workflow: Upload -> Analysis -> Visualization -> Insights"""
        
        # Step 1: Upload data file
        with open(temp_files['csv_data_csv'], 'rb') as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            response = test_client.post(
                "/api/v1/files/upload",
                files=files,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        upload_result = response.json()
        dataset_id = upload_result['dataset_id']
        
        # Step 2: Request data analysis
        analysis_request = {
            "prompt": "Analyze the uploaded dataset and provide insights",
            "dataset_id": dataset_id,
            "agent_type": "data_scientist"
        }
        
        with patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude:
            mock_claude.messages.create.return_value = Mock(
                content=[Mock(text="Dataset contains 100 records with 5 features. Key insights: Strong correlation between value and category.")]
            )
            
            response = test_client.post(
                "/api/v1/agents/process",
                json=analysis_request,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        analysis_result = response.json()
        assert analysis_result['status'] == 'success'
        
        # Step 3: Generate visualizations
        viz_request = {
            "dataset_id": dataset_id,
            "chart_type": "auto",
            "prompt": "Create visualizations for the analyzed data"
        }
        
        response = test_client.post(
            "/api/v1/viz/generate",
            json=viz_request,
            headers=test_user_token
        )
        
        assert response.status_code == 200
        viz_result = response.json()
        assert 'charts' in viz_result
        
        # Step 4: Query insights with natural language
        qa_request = {
            "question": "What are the main patterns in the data?",
            "dataset_id": dataset_id
        }
        
        with patch('scrollintel.engines.scroll_qa_engine.openai') as mock_openai:
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="The main patterns show strong correlation between categories and values."))]
            )
            
            response = test_client.post(
                "/api/v1/qa/query",
                json=qa_request,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        qa_result = response.json()
        assert qa_result['status'] == 'success'
        assert 'answer' in qa_result
    
    @pytest.mark.asyncio
    async def test_ml_model_training_workflow(self, test_client, test_user_token, temp_files, mock_ai_services):
        """Test ML model training workflow: Upload -> Train -> Evaluate -> Deploy"""
        
        # Step 1: Upload ML dataset
        with open(temp_files['ml_data_csv'], 'rb') as f:
            files = {"file": ("ml_data.csv", f, "text/csv")}
            response = test_client.post(
                "/api/v1/files/upload",
                files=files,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        dataset_id = response.json()['dataset_id']
        
        # Step 2: Train ML model
        training_request = {
            "dataset_id": dataset_id,
            "target_column": "target",
            "model_type": "classification",
            "algorithms": ["random_forest", "xgboost"]
        }
        
        response = test_client.post(
            "/api/v1/automodel/train",
            json=training_request,
            headers=test_user_token
        )
        
        assert response.status_code == 200
        training_result = response.json()
        assert training_result['status'] == 'success'
        model_id = training_result['model_id']
        
        # Step 3: Evaluate model
        response = test_client.get(
            f"/api/v1/automodel/models/{model_id}/metrics",
            headers=test_user_token
        )
        
        assert response.status_code == 200
        metrics = response.json()
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # Step 4: Make predictions
        prediction_request = {
            "model_id": model_id,
            "data": [
                {"feature_1": 0.5, "feature_2": -0.3, "feature_3": 1.2}
            ]
        }
        
        response = test_client.post(
            "/api/v1/automodel/predict",
            json=prediction_request,
            headers=test_user_token
        )
        
        assert response.status_code == 200
        predictions = response.json()
        assert 'predictions' in predictions
        assert len(predictions['predictions']) == 1
    
    @pytest.mark.asyncio
    async def test_dashboard_creation_workflow(self, test_client, test_user_token, temp_files, mock_ai_services):
        """Test dashboard creation workflow: Upload -> Analyze -> Create Dashboard -> Share"""
        
        # Step 1: Upload data
        with open(temp_files['time_series_csv'], 'rb') as f:
            files = {"file": ("time_series.csv", f, "text/csv")}
            response = test_client.post(
                "/api/v1/files/upload",
                files=files,
                headers=test_user_token
            )
        
        dataset_id = response.json()['dataset_id']
        
        # Step 2: Request BI dashboard creation
        dashboard_request = {
            "dataset_id": dataset_id,
            "dashboard_type": "executive",
            "prompt": "Create executive dashboard with key metrics and trends"
        }
        
        with patch('scrollintel.agents.scroll_bi_agent.openai') as mock_openai:
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Dashboard created with sales trends and temperature correlation"))]
            )
            
            response = test_client.post(
                "/api/v1/bi/create_dashboard",
                json=dashboard_request,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        dashboard_result = response.json()
        dashboard_id = dashboard_result['dashboard_id']
        
        # Step 3: Get dashboard configuration
        response = test_client.get(
            f"/api/v1/bi/dashboards/{dashboard_id}",
            headers=test_user_token
        )
        
        assert response.status_code == 200
        dashboard_config = response.json()
        assert 'charts' in dashboard_config
        assert 'layout' in dashboard_config
        
        # Step 4: Update dashboard in real-time
        update_request = {
            "dashboard_id": dashboard_id,
            "refresh_data": True
        }
        
        response = test_client.post(
            "/api/v1/bi/dashboards/refresh",
            json=update_request,
            headers=test_user_token
        )
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_multi_format_file_processing_workflow(self, test_client, test_user_token, temp_files):
        """Test processing multiple file formats in sequence"""
        
        file_formats = [
            ('csv_data_csv', 'text/csv'),
            ('csv_data_xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
            ('json', 'application/json')
        ]
        
        uploaded_datasets = []
        
        # Upload different file formats
        for file_key, content_type in file_formats:
            with open(temp_files[file_key], 'rb') as f:
                files = {"file": (f"test_{file_key}", f, content_type)}
                response = test_client.post(
                    "/api/v1/files/upload",
                    files=files,
                    headers=test_user_token
                )
            
            assert response.status_code == 200
            result = response.json()
            uploaded_datasets.append(result['dataset_id'])
        
        # Verify all datasets are accessible
        for dataset_id in uploaded_datasets:
            response = test_client.get(
                f"/api/v1/datasets/{dataset_id}",
                headers=test_user_token
            )
            assert response.status_code == 200
            
            dataset_info = response.json()
            assert 'schema' in dataset_info
            assert 'row_count' in dataset_info
    
    @pytest.mark.asyncio
    async def test_ai_engineer_rag_workflow(self, test_client, test_user_token, mock_ai_services):
        """Test AI Engineer RAG workflow: Index -> Query -> Generate"""
        
        # Step 1: Index documents for RAG
        documents = [
            {"id": "doc1", "content": "Machine learning best practices include data validation and model monitoring."},
            {"id": "doc2", "content": "Deep learning requires large datasets and GPU acceleration for training."},
            {"id": "doc3", "content": "MLOps involves continuous integration and deployment of ML models."}
        ]
        
        index_request = {
            "documents": documents,
            "index_name": "ml_knowledge"
        }
        
        with patch('scrollintel.agents.scroll_ai_engineer.pinecone') as mock_pinecone:
            mock_pinecone.upsert.return_value = {"upserted_count": 3}
            
            response = test_client.post(
                "/api/v1/ai_engineer/index",
                json=index_request,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        
        # Step 2: Query with RAG
        rag_request = {
            "question": "What are machine learning best practices?",
            "index_name": "ml_knowledge",
            "use_rag": True
        }
        
        with patch('scrollintel.agents.scroll_ai_engineer.pinecone') as mock_pinecone, \
             patch('scrollintel.agents.scroll_ai_engineer.openai') as mock_openai:
            
            mock_pinecone.query.return_value = {
                'matches': [
                    {'id': 'doc1', 'score': 0.9, 'metadata': {'text': documents[0]['content']}}
                ]
            }
            
            mock_openai.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Based on the knowledge base, ML best practices include data validation and model monitoring."))]
            )
            
            response = test_client.post(
                "/api/v1/ai_engineer/rag_query",
                json=rag_request,
                headers=test_user_token
            )
        
        assert response.status_code == 200
        rag_result = response.json()
        assert rag_result['status'] == 'success'
        assert 'answer' in rag_result
        assert 'sources' in rag_result
    
    @pytest.mark.asyncio
    async def test_forecasting_workflow(self, test_client, test_user_token, temp_files, mock_ai_services):
        """Test time series forecasting workflow"""
        
        # Step 1: Upload time series data
        with open(temp_files['time_series_csv'], 'rb') as f:
            files = {"file": ("time_series.csv", f, "text/csv")}
            response = test_client.post(
                "/api/v1/files/upload",
                files=files,
                headers=test_user_token
            )
        
        dataset_id = response.json()['dataset_id']
        
        # Step 2: Create forecast
        forecast_request = {
            "dataset_id": dataset_id,
            "target_column": "sales",
            "date_column": "timestamp",
            "forecast_periods": 30,
            "model_type": "prophet"
        }
        
        response = test_client.post(
            "/api/v1/forecast/create",
            json=forecast_request,
            headers=test_user_token
        )
        
        assert response.status_code == 200
        forecast_result = response.json()
        assert forecast_result['status'] == 'success'
        forecast_id = forecast_result['forecast_id']
        
        # Step 3: Get forecast results
        response = test_client.get(
            f"/api/v1/forecast/{forecast_id}/results",
            headers=test_user_token
        )
        
        assert response.status_code == 200
        results = response.json()
        assert 'predictions' in results
        assert 'confidence_intervals' in results
        
        # Step 4: Visualize forecast
        viz_request = {
            "forecast_id": forecast_id,
            "include_history": True,
            "chart_type": "line"
        }
        
        response = test_client.post(
            "/api/v1/forecast/visualize",
            json=viz_request,
            headers=test_user_token
        )
        
        assert response.status_code == 200
        viz_result = response.json()
        assert 'chart_config' in viz_result
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, test_client, test_user_token, temp_files):
        """Test error handling and recovery in workflows"""
        
        # Step 1: Upload invalid file
        invalid_content = b"This is not a valid CSV file content"
        files = {"file": ("invalid.csv", io.BytesIO(invalid_content), "text/csv")}
        
        response = test_client.post(
            "/api/v1/files/upload",
            files=files,
            headers=test_user_token
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 422]  # Bad request or validation error
        error_result = response.json()
        assert 'error' in error_result or 'detail' in error_result
        
        # Step 2: Try with valid file after error
        with open(temp_files['csv_data_csv'], 'rb') as f:
            files = {"file": ("valid.csv", f, "text/csv")}
            response = test_client.post(
                "/api/v1/files/upload",
                files=files,
                headers=test_user_token
            )
        
        # Should succeed after previous error
        assert response.status_code == 200
        
        # Step 3: Test agent error recovery
        analysis_request = {
            "prompt": "Analyze non-existent dataset",
            "dataset_id": "invalid-id",
            "agent_type": "data_scientist"
        }
        
        response = test_client.post(
            "/api/v1/agents/process",
            json=analysis_request,
            headers=test_user_token
        )
        
        # Should handle missing dataset gracefully
        assert response.status_code in [400, 404]
    
    @pytest.mark.asyncio
    async def test_concurrent_user_workflow(self, test_client, test_user_token, temp_files, mock_ai_services):
        """Test concurrent workflows from multiple users"""
        
        # Simulate multiple concurrent uploads and analyses
        async def user_workflow(user_suffix: str):
            # Upload data
            with open(temp_files['csv_data_csv'], 'rb') as f:
                files = {"file": (f"data_{user_suffix}.csv", f, "text/csv")}
                upload_response = test_client.post(
                    "/api/v1/files/upload",
                    files=files,
                    headers=test_user_token
                )
            
            if upload_response.status_code != 200:
                return {"error": "Upload failed"}
            
            dataset_id = upload_response.json()['dataset_id']
            
            # Request analysis
            analysis_request = {
                "prompt": f"Analyze dataset for user {user_suffix}",
                "dataset_id": dataset_id,
                "agent_type": "data_scientist"
            }
            
            with patch('scrollintel.agents.scroll_data_scientist.anthropic') as mock_claude:
                mock_claude.messages.create.return_value = Mock(
                    content=[Mock(text=f"Analysis complete for user {user_suffix}")]
                )
                
                analysis_response = test_client.post(
                    "/api/v1/agents/process",
                    json=analysis_request,
                    headers=test_user_token
                )
            
            return {
                "user": user_suffix,
                "upload_status": upload_response.status_code,
                "analysis_status": analysis_response.status_code,
                "dataset_id": dataset_id
            }
        
        # Run concurrent workflows
        tasks = [user_workflow(f"user_{i}") for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all workflows completed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent workflow failed: {result}")
            
            assert result['upload_status'] == 200
            assert result['analysis_status'] == 200
            assert 'dataset_id' in result