"""
Unit tests for ScrollDataScientist agent.
Tests data science workflows and statistical analysis capabilities.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.agents.scroll_data_scientist import (
    ScrollDataScientist, AnalysisType, StatisticalTest, 
    EDAReport, StatisticalTestResult, FeatureEngineeringResult
)
from scrollintel.core.interfaces import AgentRequest, AgentType, ResponseStatus


class TestScrollDataScientist:
    """Test suite for ScrollDataScientist agent"""
    
    @pytest.fixture
    def agent(self):
        """Create ScrollDataScientist agent instance"""
        return ScrollDataScientist()
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
            'experience': np.random.randint(0, 40, 100),
            'target': np.random.choice([0, 1], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample agent request"""
        return AgentRequest(
            id="test-request-1",
            user_id="test-user",
            agent_id="scroll-data-scientist",
            prompt="Perform exploratory data analysis",
            context={},
            priority=1,
            created_at=datetime.now()
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.agent_id == "scroll-data-scientist"
        assert agent.name == "ScrollDataScientist Agent"
        assert agent.agent_type == AgentType.DATA_SCIENTIST
        assert len(agent.capabilities) == 5
        
        # Check capabilities
        capability_names = [cap.name for cap in agent.capabilities]
        expected_capabilities = [
            "exploratory_data_analysis",
            "statistical_analysis", 
            "feature_engineering",
            "data_preprocessing",
            "automodel_integration"
        ]
        for expected in expected_capabilities:
            assert expected in capability_names
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test agent health check"""
        result = await agent.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_eda_request_processing(self, agent, sample_request, sample_dataset):
        """Test EDA request processing"""
        # Mock dataset loading
        with patch.object(agent, '_load_dataset', return_value=sample_dataset):
            sample_request.prompt = "perform exploratory data analysis"
            sample_request.context = {"dataset_path": "test.csv"}
            
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Exploratory Data Analysis Report" in response.content
            assert "Dataset Overview" in response.content
            assert "Summary Statistics" in response.content
            assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_request(self, agent, sample_request, sample_dataset):
        """Test statistical analysis request processing"""
        with patch.object(agent, '_load_dataset', return_value=sample_dataset):
            sample_request.prompt = "perform statistical analysis"
            sample_request.context = {"dataset_path": "test.csv"}
            
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Statistical Analysis Report" in response.content
            assert "Test Results" in response.content
    
    @pytest.mark.asyncio
    async def test_feature_engineering_request(self, agent, sample_request, sample_dataset):
        """Test feature engineering request processing"""
        with patch.object(agent, '_load_dataset', return_value=sample_dataset):
            sample_request.prompt = "perform feature engineering"
            sample_request.context = {
                "dataset_path": "test.csv",
                "target_column": "target"
            }
            
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Feature Engineering Report" in response.content
            assert "Engineered Features" in response.content
    
    @pytest.mark.asyncio
    async def test_data_preprocessing_request(self, agent, sample_request, sample_dataset):
        """Test data preprocessing request processing"""
        with patch.object(agent, '_load_dataset', return_value=sample_dataset):
            sample_request.prompt = "preprocess data"
            sample_request.context = {"dataset_path": "test.csv"}
            
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Data Preprocessing Report" in response.content
            assert "Quality Metrics" in response.content
    
    @pytest.mark.asyncio
    async def test_automodel_integration_request(self, agent, sample_request, sample_dataset):
        """Test AutoModel integration request processing"""
        with patch.object(agent, '_load_dataset', return_value=sample_dataset):
            sample_request.prompt = "prepare for model training"
            sample_request.context = {
                "dataset_path": "test.csv",
                "target_column": "target"
            }
            
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "AutoModel Integration Report" in response.content
            assert "Recommended Algorithms" in response.content
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent, sample_request):
        """Test error handling in request processing"""
        # Test with invalid dataset path
        sample_request.context = {"dataset_path": "nonexistent.csv"}
        
        response = await agent.process_request(sample_request)
        
        assert response.status == ResponseStatus.ERROR
        assert "Error processing data science request" in response.content
        assert response.error_message is not None
    
    @pytest.mark.asyncio
    async def test_load_dataset_csv(self, agent, tmp_path):
        """Test CSV dataset loading"""
        # Create temporary CSV file
        csv_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        test_data.to_csv(csv_file, index=False)
        
        result = await agent._load_dataset(str(csv_file))
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        assert list(result.columns) == ['A', 'B']
    
    @pytest.mark.asyncio
    async def test_load_dataset_unsupported_format(self, agent):
        """Test loading unsupported file format"""
        with pytest.raises(Exception) as exc_info:
            await agent._load_dataset("test.txt")
        
        assert "Unsupported file format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_eda_report(self, agent, sample_dataset):
        """Test EDA report generation"""
        report = await agent._generate_eda_report(sample_dataset)
        
        assert isinstance(report, EDAReport)
        assert 'shape' in report.dataset_info
        assert 'numerical' in report.summary_statistics
        assert 'categorical' in report.summary_statistics
        assert 'total_missing' in report.missing_values
        assert len(report.insights) > 0
        assert len(report.recommendations) > 0
    
    def test_find_high_correlations(self, agent, sample_dataset):
        """Test high correlation detection"""
        # Create correlation matrix
        numerical_data = sample_dataset.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()
        
        high_corr = agent._find_high_correlations(corr_matrix, threshold=0.5)
        
        assert isinstance(high_corr, list)
        # Each item should be a tuple of (var1, var2, correlation)
        for item in high_corr:
            assert len(item) == 3
            assert isinstance(item[2], (int, float))
    
    def test_generate_eda_insights(self, agent, sample_dataset):
        """Test EDA insights generation"""
        dataset_info = {
            'shape': sample_dataset.shape,
            'columns': {col: {'dtype': str(sample_dataset[col].dtype), 'unique_values': sample_dataset[col].nunique()} 
                       for col in sample_dataset.columns}
        }
        summary_stats = {'numerical': {}, 'categorical': {}}
        missing_values = {'total_missing': 0, 'missing_by_column': {}, 'missing_percentage': {}}
        correlations = {'high_correlations': []}
        
        insights = agent._generate_eda_insights(
            sample_dataset, dataset_info, summary_stats, missing_values, correlations
        )
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
    
    def test_generate_eda_recommendations(self, agent, sample_dataset):
        """Test EDA recommendations generation"""
        missing_values = {
            'missing_by_column': {'age': 0, 'income': 5},
            'missing_percentage': {'age': 0, 'income': 5}
        }
        correlations = {'high_correlations': [('age', 'experience', 0.8)]}
        
        recommendations = agent._generate_eda_recommendations(sample_dataset, missing_values, correlations)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_perform_statistical_tests(self, agent, sample_dataset):
        """Test statistical tests execution"""
        results = await agent._perform_statistical_tests(
            sample_dataset, "auto", [], "Test hypothesis"
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, StatisticalTestResult)
            assert hasattr(result, 'test_name')
            assert hasattr(result, 'statistic')
            assert hasattr(result, 'p_value')
            assert hasattr(result, 'significant')
    
    @pytest.mark.asyncio
    async def test_engineer_features(self, agent, sample_dataset):
        """Test feature engineering"""
        result = await agent._engineer_features(
            sample_dataset, "target", {"create_polynomial_features": True}
        )
        
        assert isinstance(result, FeatureEngineeringResult)
        assert len(result.original_features) > 0
        assert len(result.transformation_log) > 0
        assert isinstance(result.processed_data, pd.DataFrame)
        
        # Check that new features were created
        original_cols = set(sample_dataset.columns)
        processed_cols = set(result.processed_data.columns)
        new_cols = processed_cols - original_cols
        assert len(new_cols) > 0
    
    @pytest.mark.asyncio
    async def test_preprocess_data(self, agent, sample_dataset):
        """Test data preprocessing"""
        # Add some missing values and duplicates for testing
        test_data = sample_dataset.copy()
        test_data.loc[0:4, 'income'] = np.nan
        test_data = pd.concat([test_data, test_data.iloc[0:2]], ignore_index=True)
        
        processed_df, report = await agent._preprocess_data(test_data, {})
        
        assert isinstance(processed_df, pd.DataFrame)
        assert isinstance(report, dict)
        assert 'steps_applied' in report
        assert 'quality_metrics' in report
        assert 'quality_score' in report
        
        # Check that duplicates were removed
        assert processed_df.shape[0] <= test_data.shape[0]
    
    @pytest.mark.asyncio
    async def test_prepare_automodel_request(self, agent, sample_dataset):
        """Test AutoModel request preparation"""
        result = await agent._prepare_automodel_request(
            sample_dataset, "target", {}
        )
        
        assert isinstance(result, dict)
        assert 'feature_count' in result
        assert 'model_type' in result
        assert 'data_quality_score' in result
        assert 'recommended_algorithms' in result
        assert 'training_config' in result
        
        # Check model type detection
        assert result['model_type'] in ['classification', 'regression']
    
    @pytest.mark.asyncio
    async def test_claude_integration_unavailable(self, agent, sample_dataset):
        """Test Claude integration when unavailable"""
        # Mock Claude client as None
        agent.claude_client = None
        
        insights = await agent._get_claude_insights(sample_dataset, {}, "eda")
        
        assert "Claude AI integration not available" in insights
    
    @pytest.mark.asyncio
    async def test_claude_integration_with_mock(self, agent, sample_dataset):
        """Test Claude integration with mock"""
        # Mock Claude client
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock()]
        mock_message.content[0].text = "Mock AI insights"
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        
        agent.claude_client = mock_client
        
        insights = await agent._get_claude_insights(sample_dataset, {}, "eda")
        
        assert insights == "Mock AI insights"
        mock_client.messages.create.assert_called_once()
    
    def test_format_column_info(self, agent):
        """Test column information formatting"""
        columns = {
            'age': {'dtype': 'int64', 'unique_values': 50},
            'name': {'dtype': 'object', 'unique_values': 100}
        }
        
        result = agent._format_column_info(columns)
        
        assert isinstance(result, str)
        assert 'age' in result
        assert 'name' in result
        assert 'int64' in result
        assert 'object' in result
    
    def test_format_summary_stats(self, agent):
        """Test summary statistics formatting"""
        stats = {
            'age': {'mean': 35.5, 'std': 12.3, 'min': 18, 'max': 65},
            'income': {'mean': 50000, 'std': 15000, 'min': 20000, 'max': 100000}
        }
        
        result = agent._format_summary_stats(stats)
        
        assert isinstance(result, str)
        assert 'age' in result
        assert 'income' in result
        assert '35.500' in result
    
    def test_format_test_results(self, agent):
        """Test statistical test results formatting"""
        results = [
            StatisticalTestResult(
                test_name="T-Test",
                statistic=2.5,
                p_value=0.012,
                critical_value=1.96,
                interpretation="Significant difference",
                significant=True,
                effect_size=0.8
            )
        ]
        
        result = agent._format_test_results(results)
        
        assert isinstance(result, str)
        assert 'T-Test' in result
        assert '2.5000' in result
        assert '0.0120' in result
        assert 'Significant difference' in result
    
    def test_format_feature_importance(self, agent):
        """Test feature importance formatting"""
        importance = {
            'age': 0.25,
            'income': 0.35,
            'experience': 0.15,
            'education_encoded': 0.25
        }
        
        result = agent._format_feature_importance(importance)
        
        assert isinstance(result, str)
        assert 'income' in result  # Should be first (highest importance)
        assert '0.3500' in result
        assert '|' in result  # Table format
    
    def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        capabilities = agent.get_capabilities()
        
        assert len(capabilities) == 5
        assert all(hasattr(cap, 'name') for cap in capabilities)
        assert all(hasattr(cap, 'description') for cap in capabilities)
        assert all(hasattr(cap, 'input_types') for cap in capabilities)
        assert all(hasattr(cap, 'output_types') for cap in capabilities)
    
    def test_set_automodel_engine(self, agent):
        """Test setting AutoModel engine reference"""
        mock_engine = Mock()
        
        agent.set_automodel_engine(mock_engine)
        
        assert agent.automodel_engine == mock_engine
    
    @pytest.mark.asyncio
    async def test_create_eda_visualizations(self, agent, sample_dataset):
        """Test EDA visualization creation"""
        visualizations = await agent._create_eda_visualizations(sample_dataset)
        
        assert isinstance(visualizations, list)
        assert len(visualizations) > 0
        
        # Should include distribution plots and correlation heatmap
        viz_text = ' '.join(visualizations)
        assert 'Distribution plots' in viz_text or 'Correlation heatmap' in viz_text
    
    @pytest.mark.asyncio
    async def test_general_data_science_analysis(self, agent, sample_dataset):
        """Test general data science analysis"""
        with patch.object(agent, '_load_dataset', return_value=sample_dataset):
            result = await agent._general_data_science_analysis(
                "Analyze this dataset", {"dataset_path": "test.csv"}
            )
            
            assert isinstance(result, str)
            assert "Data Science Analysis" in result
            assert "Dataset Overview" in result
            assert str(sample_dataset.shape) in result


class TestDataScienceWorkflows:
    """Integration tests for data science workflows"""
    
    @pytest.fixture
    def agent(self):
        return ScrollDataScientist()
    
    @pytest.fixture
    def complex_dataset(self):
        """Create complex dataset for workflow testing"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customer_id': range(n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(10, 1, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'experience': np.random.randint(0, 40, n_samples),
            'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
            'satisfaction': np.random.randint(1, 6, n_samples),
            'performance': np.random.normal(3.5, 0.8, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        # Add some missing values
        df = pd.DataFrame(data)
        missing_indices = np.random.choice(df.index, size=50, replace=False)
        df.loc[missing_indices, 'income'] = np.nan
        
        return df
    
    @pytest.mark.asyncio
    async def test_complete_eda_workflow(self, agent, complex_dataset):
        """Test complete EDA workflow"""
        report = await agent._generate_eda_report(complex_dataset)
        
        # Verify comprehensive analysis
        assert report.dataset_info['shape'] == complex_dataset.shape
        assert len(report.summary_statistics['numerical']) > 0
        assert len(report.summary_statistics['categorical']) > 0
        assert report.missing_values['total_missing'] > 0
        assert len(report.insights) > 0
        assert len(report.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_feature_engineering_workflow(self, agent, complex_dataset):
        """Test feature engineering workflow"""
        result = await agent._engineer_features(
            complex_dataset, 
            "churn", 
            {
                "create_polynomial_features": True,
                "create_interactions": True,
                "create_bins": True
            }
        )
        
        # Verify feature engineering results
        assert len(result.engineered_features) > 0
        assert result.processed_data.shape[1] > complex_dataset.shape[1]
        assert len(result.transformation_log) > 0
        
        # Check specific feature types were created
        new_columns = set(result.processed_data.columns) - set(complex_dataset.columns)
        assert any('_squared' in col for col in new_columns)  # Polynomial features
        assert any('_x_' in col for col in new_columns)       # Interaction features
        assert any('_encoded' in col for col in new_columns)  # Encoded features
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_workflow(self, agent, complex_dataset):
        """Test statistical analysis workflow"""
        results = await agent._perform_statistical_tests(
            complex_dataset, "auto", [], "Employee churn analysis"
        )
        
        # Verify statistical tests were performed
        assert len(results) > 0
        
        # Check for different types of tests
        test_names = [result.test_name for result in results]
        assert any('Normality' in name for name in test_names)
        assert any('Correlation' in name for name in test_names)
        
        # Verify test result structure
        for result in results:
            assert isinstance(result.statistic, (int, float))
            assert isinstance(result.p_value, (int, float))
            assert isinstance(result.significant, bool)
            assert isinstance(result.interpretation, str)
    
    @pytest.mark.asyncio
    async def test_preprocessing_workflow(self, agent, complex_dataset):
        """Test data preprocessing workflow"""
        processed_df, report = await agent._preprocess_data(
            complex_dataset, 
            {"remove_outliers": True}
        )
        
        # Verify preprocessing results
        assert processed_df.shape[0] <= complex_dataset.shape[0]  # May remove outliers/duplicates
        assert 'steps_applied' in report
        assert 'quality_score' in report
        assert report['quality_score'] > 0
        
        # Check quality improvements
        original_missing = complex_dataset.isnull().sum().sum()
        processed_missing = processed_df.isnull().sum().sum()
        # Missing values should be handled (though not necessarily removed)
        assert processed_missing <= original_missing
    
    @pytest.mark.asyncio
    async def test_automodel_preparation_workflow(self, agent, complex_dataset):
        """Test AutoModel preparation workflow"""
        request = await agent._prepare_automodel_request(
            complex_dataset, "churn", {"performance_target": "accuracy"}
        )
        
        # Verify AutoModel request preparation
        assert request['model_type'] == 'classification'  # Binary target
        assert request['feature_count'] > 0
        assert request['data_quality_score'] > 0
        assert len(request['recommended_algorithms']) > 0
        assert 'training_config' in request
        
        # Check algorithm recommendations
        algorithms = request['recommended_algorithms']
        assert isinstance(algorithms, dict)
        assert len(algorithms) > 0
        
        # Verify training configuration
        config = request['training_config']
        assert config['target_column'] == 'churn'
        assert 'algorithms' in config
        assert config['test_size'] == 0.2


if __name__ == "__main__":
    pytest.main([__file__])