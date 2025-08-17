"""
ScrollIntel Data Scientist Agent
Advanced data science capabilities with AI-enhanced insights
"""

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus
import asyncio
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Claude integration
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

class AnalysisType(Enum):
    EXPLORATORY = "exploratory"
    STATISTICAL = "statistical"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    FEATURE_ENGINEERING = "feature_engineering"
    PREPROCESSING = "preprocessing"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

class StatisticalTest(Enum):
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    CORRELATION = "correlation"
    ANOVA = "anova"

@dataclass
class EDAReport:
    dataset_info: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    missing_values: Dict[str, Any]
    correlations: Dict[str, Any]
    visualizations: List[str]
    insights: List[str]
    recommendations: List[str]

@dataclass
class StatisticalTestResult:
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    interpretation: str
    significant: bool
    effect_size: Optional[float]

@dataclass
class FeatureEngineeringResult:
    original_features: List[str]
    engineered_features: List[str]
    feature_importance: Dict[str, float]
    transformation_log: List[str]
    processed_data: pd.DataFrame

class ScrollDataScientist(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="scroll-data-scientist",
            name="ScrollDataScientist Agent",
            agent_type=AgentType.DATA_SCIENTIST
        )
        self.capabilities = [
            AgentCapability(
                name="exploratory_data_analysis",
                description="Automated EDA with comprehensive statistical analysis and visualizations",
                input_types=["dataset", "csv_file", "dataframe"],
                output_types=["eda_report", "visualizations", "insights"]
            ),
            AgentCapability(
                name="statistical_analysis",
                description="Advanced statistical analysis including hypothesis testing and modeling",
                input_types=["dataset", "hypothesis", "variables"],
                output_types=["statistical_results", "test_results", "interpretation"]
            ),
            AgentCapability(
                name="feature_engineering",
                description="Automated feature engineering and data preprocessing pipelines",
                input_types=["dataset", "target_variable", "feature_config"],
                output_types=["engineered_features", "preprocessing_pipeline", "feature_importance"]
            ),
            AgentCapability(
                name="data_preprocessing",
                description="Comprehensive data cleaning and preprocessing workflows",
                input_types=["raw_dataset", "preprocessing_config"],
                output_types=["cleaned_data", "preprocessing_report", "quality_metrics"]
            ),
            AgentCapability(
                name="automodel_integration",
                description="Integration with AutoModel engine for ML model training requests",
                input_types=["processed_data", "model_requirements"],
                output_types=["model_training_request", "model_recommendations"]
            )
        ]
        
        # Initialize Claude client for AI-enhanced analysis
        if HAS_ANTHROPIC:
            self.claude_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            self.claude_client = None
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('Agg')
        sns.set_style("whitegrid")
        
        # Initialize AutoModel engine reference
        self.automodel_engine = None
        
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        try:
            # Parse the request to determine analysis type
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "eda" in prompt or "exploratory" in prompt or "explore" in prompt:
                content = await self._perform_eda(request.prompt, context)
            elif "statistical" in prompt or "hypothesis" in prompt or "test" in prompt:
                content = await self._perform_statistical_analysis(request.prompt, context)
            elif "feature" in prompt or "engineering" in prompt:
                content = await self._perform_feature_engineering(request.prompt, context)
            elif "preprocess" in prompt or "clean" in prompt:
                content = await self._perform_data_preprocessing(request.prompt, context)
            elif "model" in prompt or "train" in prompt or "automodel" in prompt:
                content = await self._integrate_with_automodel(request.prompt, context)
            else:
                content = await self._general_data_science_analysis(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"ds-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"ds-{uuid4()}",
                request_id=request.id,
                content=f"Error processing data science request: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _perform_eda(self, prompt: str, context: Dict[str, Any]) -> str:
        """Perform comprehensive exploratory data analysis with AI insights"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, list):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, pd.DataFrame):
                df = dataset
            else:
                return "Error: Unsupported dataset format. Please provide a dictionary, list, or DataFrame."
        else:
            return "Error: No dataset provided. Please provide dataset_path or dataset in context."
        
        # Generate EDA report
        eda_report = await self._generate_eda_report(df)
        
        # Get AI-enhanced insights using Claude
        ai_insights = await self._get_claude_insights(df, eda_report, "eda")
        
        # Format comprehensive EDA report
        report = f"""
# Exploratory Data Analysis Report

## Dataset Overview
- **Shape**: {eda_report.dataset_info['shape']}
- **Memory Usage**: {eda_report.dataset_info['memory_usage']} MB
- **Data Types**: {len(eda_report.dataset_info['dtypes'])} unique types

### Column Information
{self._format_column_info(eda_report.dataset_info['columns'])}

## Summary Statistics

### Numerical Variables
{self._format_summary_stats(eda_report.summary_statistics['numerical'])}

### Categorical Variables
{self._format_categorical_stats(eda_report.summary_statistics['categorical'])}

## Data Quality Assessment

### Missing Values
{self._format_missing_values(eda_report.missing_values)}

### Correlation Analysis
{self._format_correlation_analysis(eda_report.correlations)}

## Key Insights
{chr(10).join(f"- {insight}" for insight in eda_report.insights)}

## AI-Enhanced Analysis
{ai_insights}

## Recommendations
{chr(10).join(f"- {rec}" for rec in eda_report.recommendations)}

## Visualizations Generated
{chr(10).join(f"- {viz}" for viz in eda_report.visualizations)}

---
*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _perform_statistical_analysis(self, prompt: str, context: Dict[str, Any]) -> str:
        """Perform statistical analysis and hypothesis testing"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        test_type = context.get("test_type", "auto")
        variables = context.get("variables", [])
        hypothesis = context.get("hypothesis", "")
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, list):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, pd.DataFrame):
                df = dataset
            else:
                return "Error: Unsupported dataset format."
        else:
            return "Error: No dataset provided."
        
        # Perform statistical tests
        test_results = await self._perform_statistical_tests(df, test_type, variables, hypothesis)
        
        # Get AI interpretation
        ai_interpretation = await self._get_claude_insights(df, test_results, "statistical")
        
        report = f"""
# Statistical Analysis Report

## Hypothesis
{hypothesis if hypothesis else "Automated statistical analysis"}

## Test Results
{self._format_test_results(test_results)}

## AI-Enhanced Interpretation
{ai_interpretation}

## Statistical Summary
- **Tests Performed**: {len(test_results)}
- **Significant Results**: {sum(1 for result in test_results if result.significant)}
- **Confidence Level**: 95%

## Recommendations
{self._generate_statistical_recommendations(test_results)}

---
*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _perform_feature_engineering(self, prompt: str, context: Dict[str, Any]) -> str:
        """Perform automated feature engineering"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        target_column = context.get("target_column")
        feature_config = context.get("feature_config", {})
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, list):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, pd.DataFrame):
                df = dataset
            else:
                return "Error: Unsupported dataset format."
        else:
            return "Error: No dataset provided."
        
        # Perform feature engineering
        fe_result = await self._engineer_features(df, target_column, feature_config)
        
        # Get AI recommendations
        ai_recommendations = await self._get_claude_insights(df, fe_result, "feature_engineering")
        
        report = f"""
# Feature Engineering Report

## Original Dataset
- **Features**: {len(fe_result.original_features)}
- **Target**: {target_column or "Not specified"}

## Engineered Features
- **New Features Created**: {len(fe_result.engineered_features)}
- **Total Features**: {len(fe_result.processed_data.columns)}

### Feature Importance Ranking
{self._format_feature_importance(fe_result.feature_importance)}

## Transformation Log
{chr(10).join(f"- {transform}" for transform in fe_result.transformation_log)}

## AI-Enhanced Recommendations
{ai_recommendations}

## Next Steps
1. Validate engineered features with domain experts
2. Test feature combinations for model performance
3. Consider additional domain-specific transformations
4. Proceed with model training using AutoModel engine

---
*Feature engineering completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _perform_data_preprocessing(self, prompt: str, context: Dict[str, Any]) -> str:
        """Perform comprehensive data preprocessing"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        preprocessing_config = context.get("preprocessing_config", {})
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, list):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, pd.DataFrame):
                df = dataset
            else:
                return "Error: Unsupported dataset format."
        else:
            return "Error: No dataset provided."
        
        # Perform preprocessing
        processed_df, preprocessing_report = await self._preprocess_data(df, preprocessing_config)
        
        # Get AI insights on preprocessing
        ai_insights = await self._get_claude_insights(df, preprocessing_report, "preprocessing")
        
        report = f"""
# Data Preprocessing Report

## Original Dataset Quality
- **Shape**: {df.shape}
- **Missing Values**: {df.isnull().sum().sum()}
- **Duplicates**: {df.duplicated().sum()}

## Preprocessing Steps Applied
{chr(10).join(f"- {step}" for step in preprocessing_report['steps_applied'])}

## Processed Dataset Quality
- **Shape**: {processed_df.shape}
- **Missing Values**: {processed_df.isnull().sum().sum()}
- **Data Quality Score**: {round(preprocessing_report.get('quality_score', 0), 2)} out of 10

## Quality Metrics
{self._format_quality_metrics(preprocessing_report['quality_metrics'])}

## AI-Enhanced Insights
{ai_insights}

## Data Ready for Analysis
Dataset is now ready for:
- Exploratory Data Analysis
- Statistical Modeling
- Machine Learning with AutoModel engine

---
*Preprocessing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _integrate_with_automodel(self, prompt: str, context: Dict[str, Any]) -> str:
        """Integrate with AutoModel engine for ML model training"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        target_column = context.get("target_column")
        model_requirements = context.get("model_requirements", {})
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, list):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, pd.DataFrame):
                df = dataset
            else:
                return "Error: Unsupported dataset format."
        else:
            return "Error: No dataset provided."
        
        # Prepare data for AutoModel
        model_request = await self._prepare_automodel_request(df, target_column, model_requirements)
        
        # Get AI recommendations for model selection
        ai_recommendations = await self._get_claude_insights(df, model_request, "automodel")
        
        report = f"""
# AutoModel Integration Report

## Dataset Preparation
- **Features**: {model_request['feature_count']}
- **Target**: {target_column}
- **Model Type**: {model_request['model_type']}
- **Data Quality**: {round(model_request.get('data_quality_score', 0), 2)} out of 10

## Recommended Algorithms
{chr(10).join(f"- {algo}: {reason}" for algo, reason in model_request['recommended_algorithms'].items())}

## Model Training Configuration
```json
{json.dumps(model_request['training_config'], indent=2)}
```

## AI-Enhanced Model Recommendations
{ai_recommendations}

## Next Steps
1. Execute AutoModel training with recommended configuration
2. Compare model performance across algorithms
3. Select best performing model for deployment
4. Set up model monitoring and retraining pipeline

## AutoModel Request Ready
Dataset and configuration prepared for AutoModel engine training

---
*AutoModel integration prepared at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    async def _general_data_science_analysis(self, prompt: str, context: Dict[str, Any]) -> str:
        """General data science analysis using Claude AI"""
        dataset_path = context.get("dataset_path")
        dataset = context.get("dataset")
        
        if dataset_path:
            df = await self._load_dataset(dataset_path)
        elif dataset is not None:
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, list):
                df = pd.DataFrame(dataset)
            elif isinstance(dataset, pd.DataFrame):
                df = dataset
            else:
                return "Error: Unsupported dataset format."
        else:
            return "Error: No dataset provided."
        
        # Get AI-powered analysis
        ai_analysis = await self._get_claude_insights(df, {"prompt": prompt}, "general")
        
        # Basic dataset overview
        overview = f"""
# Data Science Analysis

## Dataset Overview
- **Shape**: {df.shape}
- **Columns**: {list(df.columns)}
- **Data Types**: {df.dtypes.value_counts().to_dict()}

## User Request
{prompt}

## AI-Powered Analysis
{ai_analysis}

## Quick Statistics
{df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "No numerical columns for statistics"}

---
*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return overview
    
    async def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from various file formats"""
        try:
            if dataset_path.endswith('.csv'):
                return pd.read_csv(dataset_path)
            elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
                return pd.read_excel(dataset_path)
            elif dataset_path.endswith('.json'):
                return pd.read_json(dataset_path)
            elif dataset_path.endswith('.parquet'):
                return pd.read_parquet(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    async def _generate_eda_report(self, df: pd.DataFrame) -> EDAReport:
        """Generate comprehensive EDA report"""
        
        # Dataset info
        dataset_info = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': df.dtypes.value_counts().to_dict(),
            'columns': {col: {'dtype': str(df[col].dtype), 'unique_values': df[col].nunique()} 
                       for col in df.columns}
        }
        
        # Summary statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        summary_stats = {
            'numerical': df[numerical_cols].describe().to_dict() if len(numerical_cols) > 0 else {},
            'categorical': {col: df[col].value_counts().head(10).to_dict() 
                          for col in categorical_cols} if len(categorical_cols) > 0 else {}
        }
        
        # Missing values analysis
        missing_values = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # Correlation analysis
        correlations = {}
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            correlations = {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(corr_matrix)
            }
        
        # Generate insights
        insights = self._generate_eda_insights(df, dataset_info, summary_stats, missing_values, correlations)
        
        # Generate recommendations
        recommendations = self._generate_eda_recommendations(df, missing_values, correlations)
        
        # Create visualizations
        visualizations = await self._create_eda_visualizations(df)
        
        return EDAReport(
            dataset_info=dataset_info,
            summary_statistics=summary_stats,
            missing_values=missing_values,
            correlations=correlations,
            visualizations=visualizations,
            insights=insights,
            recommendations=recommendations
        )
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find highly correlated variable pairs"""
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        return high_corr
    
    def _generate_eda_insights(self, df: pd.DataFrame, dataset_info: Dict, summary_stats: Dict, 
                              missing_values: Dict, correlations: Dict) -> List[str]:
        """Generate automated insights from EDA"""
        insights = []
        
        # Dataset size insights
        if df.shape[0] > 100000:
            insights.append("Large dataset detected - consider sampling for initial analysis")
        elif df.shape[0] < 1000:
            insights.append("Small dataset - be cautious of overfitting in modeling")
        
        # Missing data insights
        if missing_values['total_missing'] > 0:
            missing_pct = (missing_values['total_missing'] / (df.shape[0] * df.shape[1])) * 100
            if missing_pct > 20:
                insights.append(f"High missing data rate ({missing_pct:.1f}%) - data quality concerns")
            elif missing_pct > 5:
                insights.append(f"Moderate missing data ({missing_pct:.1f}%) - imputation strategies needed")
        
        # Data type insights
        if len(df.select_dtypes(include=['object']).columns) > len(df.select_dtypes(include=[np.number]).columns):
            insights.append("Categorical-heavy dataset - consider encoding strategies")
        
        # Correlation insights
        if 'high_correlations' in correlations and len(correlations['high_correlations']) > 0:
            insights.append(f"Found {len(correlations['high_correlations'])} highly correlated variable pairs")
        
        # Unique value insights
        for col, info in dataset_info['columns'].items():
            if info['unique_values'] == 1:
                insights.append(f"Column '{col}' has only one unique value - consider removing")
            elif info['unique_values'] == df.shape[0]:
                insights.append(f"Column '{col}' has all unique values - potential identifier")
        
        return insights
    
    def _generate_eda_recommendations(self, df: pd.DataFrame, missing_values: Dict, correlations: Dict) -> List[str]:
        """Generate actionable recommendations from EDA"""
        recommendations = []
        
        # Missing data recommendations
        for col, missing_count in missing_values['missing_by_column'].items():
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                if missing_pct > 50:
                    recommendations.append(f"Consider dropping column '{col}' ({missing_pct:.1f}% missing)")
                elif missing_pct > 10:
                    recommendations.append(f"Implement imputation strategy for '{col}' ({missing_pct:.1f}% missing)")
        
        # Correlation recommendations
        if 'high_correlations' in correlations:
            for var1, var2, corr_val in correlations['high_correlations']:
                recommendations.append(f"Consider feature selection between '{var1}' and '{var2}' (r={corr_val:.3f})")
        
        # Data type recommendations
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df[col].nunique() < 10:
                recommendations.append(f"Consider encoding categorical variable '{col}'")
        
        # General recommendations
        recommendations.append("Perform outlier detection on numerical variables")
        recommendations.append("Consider feature scaling for machine learning models")
        recommendations.append("Validate data quality with domain experts")
        
        return recommendations
    
    async def _create_eda_visualizations(self, df: pd.DataFrame) -> List[str]:
        """Create EDA visualizations and return descriptions"""
        visualizations = []
        
        try:
            # Distribution plots for numerical variables
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(numerical_cols[:6]):  # Limit to 6 plots
                    plt.subplot(2, 3, i+1)
                    df[col].hist(bins=30, alpha=0.7)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(f'eda_distributions_{uuid4().hex[:8]}.png', dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append("Distribution plots for numerical variables")
            
            # Correlation heatmap
            if len(numerical_cols) > 1:
                plt.figure(figsize=(12, 8))
                corr_matrix = df[numerical_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(f'eda_correlation_{uuid4().hex[:8]}.png', dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append("Correlation heatmap")
            
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                plt.figure(figsize=(12, 6))
                sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                plt.title('Missing Values Pattern')
                plt.tight_layout()
                plt.savefig(f'eda_missing_{uuid4().hex[:8]}.png', dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append("Missing values pattern heatmap")
            
        except Exception as e:
            visualizations.append(f"Error creating visualizations: {str(e)}")
        
        return visualizations
    
    async def _perform_statistical_tests(self, df: pd.DataFrame, test_type: str, 
                                       variables: List[str], hypothesis: str) -> List[StatisticalTestResult]:
        """Perform various statistical tests"""
        results = []
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        try:
            # Normality tests for numerical variables
            for col in numerical_cols[:5]:  # Limit to 5 variables
                if len(df[col].dropna()) > 3:
                    stat, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
                    results.append(StatisticalTestResult(
                        test_name=f"Shapiro-Wilk Normality Test ({col})",
                        statistic=stat,
                        p_value=p_value,
                        critical_value=None,
                        interpretation=f"Data is {'normally' if p_value > 0.05 else 'not normally'} distributed",
                        significant=p_value <= 0.05,
                        effect_size=None
                    ))
            
            # Correlation tests between numerical variables
            if len(numerical_cols) >= 2:
                for i in range(min(3, len(numerical_cols))):
                    for j in range(i+1, min(3, len(numerical_cols))):
                        col1, col2 = numerical_cols[i], numerical_cols[j]
                        clean_data = df[[col1, col2]].dropna()
                        if len(clean_data) > 3:
                            corr_coef, p_value = pearsonr(clean_data[col1], clean_data[col2])
                            results.append(StatisticalTestResult(
                                test_name=f"Pearson Correlation ({col1} vs {col2})",
                                statistic=corr_coef,
                                p_value=p_value,
                                critical_value=None,
                                interpretation=f"{'Significant' if p_value <= 0.05 else 'No significant'} correlation",
                                significant=p_value <= 0.05,
                                effect_size=abs(corr_coef)
                            ))
            
            # Chi-square tests for categorical variables
            if len(categorical_cols) >= 2:
                for i in range(min(2, len(categorical_cols))):
                    for j in range(i+1, min(2, len(categorical_cols))):
                        col1, col2 = categorical_cols[i], categorical_cols[j]
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        if contingency_table.size > 1:
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            results.append(StatisticalTestResult(
                                test_name=f"Chi-square Independence Test ({col1} vs {col2})",
                                statistic=chi2,
                                p_value=p_value,
                                critical_value=stats.chi2.ppf(0.95, dof),
                                interpretation=f"Variables are {'dependent' if p_value <= 0.05 else 'independent'}",
                                significant=p_value <= 0.05,
                                effect_size=np.sqrt(chi2 / (contingency_table.sum().sum() * min(contingency_table.shape) - 1))
                            ))
            
        except Exception as e:
            results.append(StatisticalTestResult(
                test_name="Error in statistical testing",
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                interpretation=f"Error: {str(e)}",
                significant=False,
                effect_size=None
            ))
        
        return results
    
    async def _engineer_features(self, df: pd.DataFrame, target_column: Optional[str], 
                               feature_config: Dict[str, Any]) -> FeatureEngineeringResult:
        """Perform automated feature engineering"""
        
        original_features = df.columns.tolist()
        if target_column and target_column in original_features:
            original_features.remove(target_column)
        
        processed_df = df.copy()
        transformation_log = []
        engineered_features = []
        
        try:
            # Handle missing values
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
            
            # Numerical imputation
            for col in numerical_cols:
                if processed_df[col].isnull().sum() > 0:
                    if feature_config.get('numerical_imputation', 'median') == 'mean':
                        processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                    else:
                        processed_df[col].fillna(processed_df[col].median(), inplace=True)
                    transformation_log.append(f"Imputed missing values in {col}")
            
            # Categorical imputation
            for col in categorical_cols:
                if processed_df[col].isnull().sum() > 0:
                    processed_df[col].fillna(processed_df[col].mode()[0] if len(processed_df[col].mode()) > 0 else 'Unknown', inplace=True)
                    transformation_log.append(f"Imputed missing values in {col}")
            
            # Create polynomial features for numerical variables
            if feature_config.get('create_polynomial_features', True) and len(numerical_cols) > 0:
                for col in numerical_cols[:3]:  # Limit to 3 columns
                    processed_df[f'{col}_squared'] = processed_df[col] ** 2
                    engineered_features.append(f'{col}_squared')
                    transformation_log.append(f"Created polynomial feature: {col}_squared")
            
            # Create interaction features
            if feature_config.get('create_interactions', True) and len(numerical_cols) >= 2:
                for i in range(min(2, len(numerical_cols))):
                    for j in range(i+1, min(3, len(numerical_cols))):
                        col1, col2 = numerical_cols[i], numerical_cols[j]
                        interaction_name = f'{col1}_x_{col2}'
                        processed_df[interaction_name] = processed_df[col1] * processed_df[col2]
                        engineered_features.append(interaction_name)
                        transformation_log.append(f"Created interaction feature: {interaction_name}")
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_cols:
                if processed_df[col].nunique() < 50:  # Avoid high cardinality
                    le = LabelEncoder()
                    processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
                    label_encoders[col] = le
                    engineered_features.append(f'{col}_encoded')
                    transformation_log.append(f"Label encoded categorical variable: {col}")
            
            # Create binned features for numerical variables
            if feature_config.get('create_bins', True):
                for col in numerical_cols[:2]:  # Limit to 2 columns
                    try:
                        processed_df[f'{col}_binned'] = pd.cut(processed_df[col], bins=5, labels=False)
                        engineered_features.append(f'{col}_binned')
                        transformation_log.append(f"Created binned feature: {col}_binned")
                    except Exception:
                        pass
            
            # Calculate feature importance if target is provided
            feature_importance = {}
            if target_column and target_column in processed_df.columns:
                try:
                    feature_cols = [col for col in processed_df.columns if col != target_column]
                    X = processed_df[feature_cols].select_dtypes(include=[np.number])
                    y = processed_df[target_column]
                    
                    if len(X.columns) > 0 and not y.isnull().all():
                        # Use mutual information for feature importance
                        if y.dtype in ['object', 'category'] or y.nunique() < 20:
                            # Classification
                            y_encoded = LabelEncoder().fit_transform(y.astype(str))
                            importance_scores = mutual_info_classif(X.fillna(0), y_encoded)
                        else:
                            # Regression
                            from sklearn.feature_selection import mutual_info_regression
                            importance_scores = mutual_info_regression(X.fillna(0), y)
                        
                        feature_importance = dict(zip(X.columns, importance_scores))
                        transformation_log.append("Calculated feature importance scores")
                
                except Exception as e:
                    transformation_log.append(f"Could not calculate feature importance: {str(e)}")
            
        except Exception as e:
            transformation_log.append(f"Error in feature engineering: {str(e)}")
        
        return FeatureEngineeringResult(
            original_features=original_features,
            engineered_features=engineered_features,
            feature_importance=feature_importance,
            transformation_log=transformation_log,
            processed_data=processed_df
        )
    
    async def _preprocess_data(self, df: pd.DataFrame, preprocessing_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive data preprocessing"""
        
        processed_df = df.copy()
        steps_applied = []
        quality_metrics = {}
        
        try:
            # Remove duplicates
            initial_shape = processed_df.shape
            processed_df = processed_df.drop_duplicates()
            if processed_df.shape[0] < initial_shape[0]:
                steps_applied.append(f"Removed {initial_shape[0] - processed_df.shape[0]} duplicate rows")
            
            # Handle outliers using IQR method
            if preprocessing_config.get('remove_outliers', True):
                numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_count = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
                    if outliers_count > 0:
                        processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
                        steps_applied.append(f"Removed {outliers_count} outliers from {col}")
            
            # Data type optimization
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='ignore')
                        if processed_df[col].dtype != 'object':
                            steps_applied.append(f"Converted {col} to numeric type")
                    except:
                        pass
                elif processed_df[col].dtype in ['int64', 'float64']:
                    # Optimize numeric types
                    if processed_df[col].dtype == 'int64':
                        if processed_df[col].min() >= 0 and processed_df[col].max() <= 255:
                            processed_df[col] = processed_df[col].astype('uint8')
                        elif processed_df[col].min() >= -128 and processed_df[col].max() <= 127:
                            processed_df[col] = processed_df[col].astype('int8')
                        elif processed_df[col].min() >= -32768 and processed_df[col].max() <= 32767:
                            processed_df[col] = processed_df[col].astype('int16')
                        elif processed_df[col].min() >= -2147483648 and processed_df[col].max() <= 2147483647:
                            processed_df[col] = processed_df[col].astype('int32')
            
            # Calculate quality metrics
            quality_metrics = {
                'completeness': (1 - processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1])) * 100,
                'uniqueness': (1 - processed_df.duplicated().sum() / processed_df.shape[0]) * 100,
                'consistency': 100,  # Placeholder - would need domain-specific rules
                'validity': 100,     # Placeholder - would need validation rules
            }
            
            # Overall quality score
            quality_score = np.mean(list(quality_metrics.values())) / 10
            
            preprocessing_report = {
                'steps_applied': steps_applied,
                'quality_metrics': quality_metrics,
                'quality_score': quality_score,
                'original_shape': df.shape,
                'processed_shape': processed_df.shape
            }
            
        except Exception as e:
            steps_applied.append(f"Error in preprocessing: {str(e)}")
            preprocessing_report = {
                'steps_applied': steps_applied,
                'quality_metrics': {'error': str(e)},
                'quality_score': 0,
                'original_shape': df.shape,
                'processed_shape': processed_df.shape
            }
        
        return processed_df, preprocessing_report
    
    async def _prepare_automodel_request(self, df: pd.DataFrame, target_column: Optional[str], 
                                       model_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request for AutoModel engine"""
        
        # Determine model type
        model_type = "classification"
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            if target_series.dtype in ['int64', 'float64'] and target_series.nunique() > 20:
                model_type = "regression"
        
        # Feature analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        feature_count = len(numerical_cols) + len(categorical_cols)
        
        # Data quality assessment
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        data_quality_score = max(0, 10 - missing_pct/10)  # Simple quality score
        
        # Algorithm recommendations based on data characteristics
        recommended_algorithms = {}
        
        if df.shape[0] < 1000:
            recommended_algorithms["random_forest"] = "Good for small datasets with interpretability"
        elif df.shape[0] > 100000:
            recommended_algorithms["xgboost"] = "Efficient for large datasets"
            recommended_algorithms["neural_network"] = "Can capture complex patterns in large data"
        else:
            recommended_algorithms["random_forest"] = "Balanced performance and interpretability"
            recommended_algorithms["xgboost"] = "High performance gradient boosting"
        
        if feature_count > 50:
            recommended_algorithms["neural_network"] = "Good for high-dimensional data"
        
        # Training configuration
        training_config = {
            "dataset_path": "processed_data.csv",  # Would be saved separately
            "target_column": target_column,
            "feature_columns": numerical_cols + categorical_cols,
            "model_type": model_type,
            "algorithms": list(recommended_algorithms.keys()),
            "cross_validation": True,
            "hyperparameter_tuning": True,
            "test_size": 0.2,
            "random_state": 42
        }
        
        return {
            "feature_count": feature_count,
            "model_type": model_type,
            "data_quality_score": data_quality_score,
            "recommended_algorithms": recommended_algorithms,
            "training_config": training_config,
            "dataset_shape": df.shape,
            "missing_data_pct": missing_pct
        }
    
    async def _get_claude_insights(self, df: pd.DataFrame, analysis_data: Any, analysis_type: str) -> str:
        """Get AI-enhanced insights using Claude"""
        
        if not self.claude_client:
            return "Claude AI integration not available. Install anthropic package for enhanced insights."
        
        try:
            # Prepare context for Claude
            dataset_summary = f"""
Dataset Shape: {df.shape}
Columns: {list(df.columns)}
Data Types: {df.dtypes.value_counts().to_dict()}
Missing Values: {df.isnull().sum().sum()}
"""
            
            if analysis_type == "eda":
                prompt = f"""
As a senior data scientist, analyze this dataset and provide insights:

{dataset_summary}

Key findings from automated analysis:
- Dataset has {df.shape[0]} rows and {df.shape[1]} columns
- Missing values: {df.isnull().sum().sum()}
- Numerical columns: {len(df.select_dtypes(include=[np.number]).columns)}
- Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}

Provide:
1. Key insights about data quality and structure
2. Potential data issues to investigate
3. Recommendations for next steps in analysis
4. Suggestions for feature engineering or modeling approaches

Keep response concise and actionable.
"""
            
            elif analysis_type == "statistical":
                prompt = f"""
As a statistician, interpret these statistical test results:

{dataset_summary}

Statistical tests performed: {len(analysis_data) if isinstance(analysis_data, list) else 'Multiple tests'}

Provide:
1. Interpretation of statistical significance
2. Practical significance and effect sizes
3. Assumptions and limitations
4. Recommendations for further analysis

Keep response focused on actionable insights.
"""
            
            elif analysis_type == "feature_engineering":
                prompt = f"""
As a machine learning engineer, evaluate this feature engineering:

{dataset_summary}

Features engineered: {len(analysis_data.engineered_features) if hasattr(analysis_data, 'engineered_features') else 'Multiple features'}

Provide:
1. Assessment of feature engineering quality
2. Additional feature engineering suggestions
3. Feature selection recommendations
4. Model-specific considerations

Keep response practical and implementation-focused.
"""
            
            elif analysis_type == "preprocessing":
                prompt = f"""
As a data engineer, evaluate this data preprocessing:

{dataset_summary}

Preprocessing steps applied: {len(analysis_data.get('steps_applied', [])) if isinstance(analysis_data, dict) else 'Multiple steps'}

Provide:
1. Assessment of preprocessing quality
2. Additional preprocessing recommendations
3. Data pipeline considerations
4. Quality assurance suggestions

Keep response focused on data quality and pipeline robustness.
"""
            
            elif analysis_type == "automodel":
                prompt = f"""
As an ML engineer, evaluate this AutoModel preparation:

{dataset_summary}

Model type: {analysis_data.get('model_type', 'Unknown')}
Recommended algorithms: {list(analysis_data.get('recommended_algorithms', {}).keys())}

Provide:
1. Assessment of model selection strategy
2. Additional algorithm recommendations
3. Training and validation considerations
4. Deployment and monitoring suggestions

Keep response focused on model performance and production readiness.
"""
            
            else:
                prompt = f"""
As a data scientist, analyze this dataset:

{dataset_summary}

User request: {analysis_data.get('prompt', 'General analysis')}

Provide comprehensive insights and recommendations for data analysis and modeling.
"""
            
            # Call Claude API
            response = await self._call_claude(prompt)
            return response
            
        except Exception as e:
            return f"Error getting AI insights: {str(e)}"
    
    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API for AI insights"""
        try:
            message = await self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Claude API error: {str(e)}"
    
    # Formatting helper methods
    def _format_column_info(self, columns: Dict[str, Dict]) -> str:
        """Format column information for display"""
        result = "| Column | Data Type | Unique Values |\n|--------|-----------|---------------|\n"
        for col, info in columns.items():
            result += f"| {col} | {info['dtype']} | {info['unique_values']} |\n"
        return result
    
    def _format_summary_stats(self, stats: Dict) -> str:
        """Format summary statistics for display"""
        if not stats:
            return "No numerical variables found."
        
        result = ""
        for col, col_stats in stats.items():
            result += f"\n**{col}:**\n"
            for stat, value in col_stats.items():
                result += f"- {stat}: {value:.3f}\n"
        return result
    
    def _format_categorical_stats(self, stats: Dict) -> str:
        """Format categorical statistics for display"""
        if not stats:
            return "No categorical variables found."
        
        result = ""
        for col, value_counts in stats.items():
            result += f"\n**{col} (Top values):**\n"
            for value, count in value_counts.items():
                result += f"- {value}: {count}\n"
        return result
    
    def _format_missing_values(self, missing_info: Dict) -> str:
        """Format missing values information"""
        result = f"**Total Missing Values:** {missing_info['total_missing']}\n\n"
        
        if missing_info['total_missing'] > 0:
            result += "**Missing by Column:**\n"
            for col, count in missing_info['missing_by_column'].items():
                if count > 0:
                    pct = missing_info['missing_percentage'][col]
                    result += f"- {col}: {count} ({pct:.1f}%)\n"
        
        return result
    
    def _format_correlation_analysis(self, correlations: Dict) -> str:
        """Format correlation analysis"""
        if not correlations:
            return "Insufficient numerical variables for correlation analysis."
        
        result = ""
        if 'high_correlations' in correlations and correlations['high_correlations']:
            result += "**High Correlations (|r| > 0.7):**\n"
            for var1, var2, corr_val in correlations['high_correlations']:
                result += f"- {var1} ↔ {var2}: r = {corr_val:.3f}\n"
        else:
            result += "No high correlations found (|r| > 0.7)."
        
        return result
    
    def _format_test_results(self, results: List[StatisticalTestResult]) -> str:
        """Format statistical test results"""
        if not results:
            return "No statistical tests performed."
        
        result = ""
        for test_result in results:
            result += f"\n**{test_result.test_name}**\n"
            result += f"- Statistic: {test_result.statistic:.4f}\n"
            result += f"- P-value: {test_result.p_value:.4f}\n"
            if test_result.critical_value:
                result += f"- Critical Value: {test_result.critical_value:.4f}\n"
            if test_result.effect_size:
                result += f"- Effect Size: {test_result.effect_size:.4f}\n"
            result += f"- Result: {test_result.interpretation}\n"
            result += f"- Significant: {'Yes' if test_result.significant else 'No'}\n"
        
        return result
    
    def _generate_statistical_recommendations(self, results: List[StatisticalTestResult]) -> str:
        """Generate recommendations based on statistical test results"""
        recommendations = []
        
        significant_tests = [r for r in results if r.significant]
        
        if len(significant_tests) > 0:
            recommendations.append(f"Found {len(significant_tests)} statistically significant results")
            recommendations.append("Investigate significant relationships for business insights")
        
        normality_tests = [r for r in results if "Normality" in r.test_name]
        non_normal = [r for r in normality_tests if r.significant]
        
        if len(non_normal) > 0:
            recommendations.append("Consider non-parametric tests for non-normal distributions")
            recommendations.append("Apply data transformations to achieve normality if needed")
        
        correlation_tests = [r for r in results if "Correlation" in r.test_name]
        strong_correlations = [r for r in correlation_tests if r.effect_size and r.effect_size > 0.7]
        
        if len(strong_correlations) > 0:
            recommendations.append("Strong correlations detected - consider multicollinearity in modeling")
        
        recommendations.append("Validate statistical assumptions before drawing conclusions")
        recommendations.append("Consider practical significance alongside statistical significance")
        
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _format_feature_importance(self, importance: Dict[str, float]) -> str:
        """Format feature importance scores"""
        if not importance:
            return "Feature importance not calculated."
        
        # Sort by importance score
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        result = "| Feature | Importance Score |\n|---------|------------------|\n"
        for feature, score in sorted_features[:10]:  # Top 10 features
            result += f"| {feature} | {score:.4f} |\n"
        
        return result
    
    def _format_quality_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format data quality metrics"""
        if 'error' in metrics:
            return f"Error calculating quality metrics: {metrics['error']}"
        
        result = ""
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                result += f"- **{metric.replace('_', ' ').title()}**: {value:.2f}%\n"
        
        return result
    
    def get_capabilities(self) -> List[AgentCapability]:
        return self.capabilities
    
    async def health_check(self) -> bool:
        return True
    
    def set_automodel_engine(self, automodel_engine):
        """Set reference to AutoModel engine for integration"""
        self.automodel_engine = automodel_engine