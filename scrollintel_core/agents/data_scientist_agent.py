"""
Data Scientist Agent - Data analysis and insights generation
Enhanced with comprehensive statistical analysis and pattern detection
"""
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from .base import Agent, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


class DataScientistAgent(Agent):
    """Data Scientist Agent for exploratory data analysis and insights"""
    
    def __init__(self):
        super().__init__(
            name="Data Scientist Agent",
            description="Provides comprehensive exploratory data analysis, statistical insights, data quality assessment, correlation analysis, and pattern detection"
        )
        self.data_cache = {}  # Cache for processed datasets
        self.analysis_cache = {}  # Cache for analysis results
    
    def get_capabilities(self) -> List[str]:
        """Return Data Scientist agent capabilities"""
        return [
            "Exploratory data analysis (EDA)",
            "Comprehensive statistical analysis",
            "Data quality assessment and recommendations",
            "Correlation analysis (Pearson, Spearman, Chi-square)",
            "Pattern detection and clustering",
            "Outlier detection and anomaly analysis",
            "Feature importance analysis",
            "Data visualization recommendations",
            "Automated insights generation",
            "Data profiling and schema analysis",
            "Missing value analysis and recommendations",
            "Distribution analysis and normality testing",
            "Hypothesis testing and statistical significance",
            "Dimensionality reduction (PCA)",
            "Time series pattern detection"
        ]
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process data science requests with comprehensive analysis"""
        start_time = time.time()
        
        try:
            query = request.query.lower()
            context = request.context
            parameters = request.parameters
            
            # Check if data is provided in context
            data = self._extract_data_from_context(context)
            analysis_type = self._classify_analysis_type(query)
            
            # Route to appropriate analysis method
            if data is not None:
                # Data is available - perform actual analysis
                if "quality" in query or "assess" in query:
                    result = await self._assess_data_quality_with_data(data, parameters)
                elif "correlation" in query or "relationship" in query:
                    result = await self._analyze_correlations_with_data(data, parameters)
                elif "pattern" in query or "cluster" in query:
                    result = await self._detect_patterns_with_data(data, parameters)
                elif "outlier" in query or "anomaly" in query:
                    result = self._detect_outliers_with_data(data, parameters)
                elif "insights" in query or "summary" in query:
                    result = await self._generate_comprehensive_insights(data, parameters)
                elif "profile" in query or "describe" in query:
                    result = await self._profile_data(data, parameters)
                else:
                    result = await self._perform_comprehensive_analysis(data, parameters)
            else:
                # No data provided - give guidance and recommendations
                if "quality" in query:
                    result = self._provide_quality_guidance()
                elif "correlation" in query:
                    result = self._provide_correlation_guidance()
                elif "pattern" in query:
                    result = self._provide_pattern_guidance()
                elif "insights" in query:
                    result = self._provide_insights_guidance()
                else:
                    result = self._provide_general_guidance(request.query)
            
            # Generate visualization recommendations
            viz_recommendations = self._generate_visualization_recommendations(
                data, analysis_type, result if data is not None else None
            )
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                result=result,
                metadata={
                    "analysis_type": analysis_type,
                    "data_available": data is not None,
                    "data_shape": data.shape if data is not None else None,
                    "visualization_recommendations": viz_recommendations,
                    "confidence_score": self._calculate_confidence_score(data, result)
                },
                processing_time=time.time() - start_time,
                confidence_score=self._calculate_confidence_score(data, result),
                suggestions=self._generate_suggestions(query, data, result)
            )
            
        except Exception as e:
            logger.error(f"Data Scientist Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e),
                error_code="DATA_ANALYSIS_ERROR",
                processing_time=time.time() - start_time,
                suggestions=[
                    "Please ensure your data is properly formatted",
                    "Try uploading a CSV, Excel, or JSON file",
                    "Check that your data contains numeric columns for statistical analysis"
                ]
            )
    
    def _extract_data_from_context(self, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract and prepare data from request context"""
        try:
            # Check for various data formats in context
            if "dataframe" in context:
                return context["dataframe"]
            elif "data" in context:
                data = context["data"]
                if isinstance(data, pd.DataFrame):
                    return data
                elif isinstance(data, dict):
                    return pd.DataFrame(data)
                elif isinstance(data, list):
                    return pd.DataFrame(data)
            elif "file_path" in context:
                # Load data from file path
                file_path = context["file_path"]
                if file_path.endswith('.csv'):
                    return pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    return pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    return pd.read_json(file_path)
            elif "dataset_id" in context:
                # Check cache for dataset
                dataset_id = context["dataset_id"]
                if dataset_id in self.data_cache:
                    return self.data_cache[dataset_id]
            
            return None
        except Exception as e:
            logger.error(f"Error extracting data from context: {e}")
            return None

    async def _perform_comprehensive_analysis(self, data: pd.DataFrame, 
                                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive data analysis on actual data"""
        try:
            analysis_results = {
                "data_overview": self._get_data_overview(data),
                "statistical_summary": self._get_statistical_summary(data),
                "data_quality": self._assess_data_quality_metrics(data),
                "correlation_analysis": self._perform_correlation_analysis(data),
                "distribution_analysis": self._analyze_distributions(data),
                "outlier_analysis": self._detect_outliers(data),
                "pattern_analysis": self._detect_basic_patterns(data),
                "insights": self._extract_automated_insights(data),
                "recommendations": self._generate_analysis_recommendations(data)
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": str(e), "analysis_type": "comprehensive_failed"}

    def _get_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview of the dataset"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            "shape": {"rows": len(data), "columns": len(data.columns)},
            "columns": {
                "total": len(data.columns),
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols)
            },
            "column_details": {
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": datetime_cols
            },
            "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "data_types": data.dtypes.astype(str).to_dict()
        }

    def _get_statistical_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {"message": "No numeric columns found for statistical analysis"}
        
        summary = {
            "descriptive_statistics": numeric_data.describe().to_dict(),
            "additional_statistics": {}
        }
        
        # Add additional statistics for each numeric column
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                summary["additional_statistics"][col] = {
                    "skewness": float(stats.skew(col_data)),
                    "kurtosis": float(stats.kurtosis(col_data)),
                    "variance": float(col_data.var()),
                    "coefficient_of_variation": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0,
                    "range": float(col_data.max() - col_data.min()),
                    "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25))
                }
        
        return summary
    
    async def _assess_data_quality_with_data(self, data: pd.DataFrame, 
                                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data quality assessment with actual data"""
        try:
            quality_metrics = self._assess_data_quality_metrics(data)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(quality_metrics)
            
            # Generate specific recommendations
            recommendations = self._generate_quality_recommendations(quality_metrics, data)
            
            return {
                "quality_score": quality_score,
                "quality_metrics": quality_metrics,
                "recommendations": recommendations,
                "priority_issues": self._identify_priority_issues(quality_metrics),
                "data_health_summary": self._generate_health_summary(quality_metrics, quality_score)
            }
            
        except Exception as e:
            logger.error(f"Error in data quality assessment: {e}")
            return {"error": str(e), "analysis_type": "quality_assessment_failed"}

    def _assess_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed data quality metrics"""
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        
        quality_metrics = {
            "completeness": {
                "missing_values_total": int(missing_cells),
                "missing_percentage": float(missing_cells / total_cells * 100),
                "missing_by_column": data.isnull().sum().to_dict(),
                "missing_percentage_by_column": (data.isnull().sum() / len(data) * 100).to_dict()
            },
            "uniqueness": {
                "duplicate_rows": int(data.duplicated().sum()),
                "duplicate_percentage": float(data.duplicated().sum() / len(data) * 100),
                "unique_values_by_column": {col: int(data[col].nunique()) for col in data.columns}
            },
            "consistency": self._check_data_consistency(data),
            "validity": self._check_data_validity(data)
        }
        
        return quality_metrics

    def _check_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency issues"""
        consistency_issues = {
            "mixed_data_types": [],
            "inconsistent_formats": [],
            "case_inconsistencies": []
        }
        
        for col in data.columns:
            # Check for mixed data types in object columns
            if data[col].dtype == 'object':
                sample_values = data[col].dropna().head(100)
                types_found = set(type(val).__name__ for val in sample_values)
                if len(types_found) > 1:
                    consistency_issues["mixed_data_types"].append({
                        "column": col,
                        "types_found": list(types_found)
                    })
                
                # Check for case inconsistencies in string data
                if sample_values.dtype == 'object':
                    unique_values = sample_values.unique()
                    if len(unique_values) > 1:
                        lower_values = [str(val).lower() for val in unique_values]
                        if len(set(lower_values)) < len(unique_values):
                            consistency_issues["case_inconsistencies"].append(col)
        
        return consistency_issues

    def _check_data_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity issues"""
        validity_issues = {
            "negative_values_in_positive_columns": [],
            "extreme_outliers": [],
            "invalid_ranges": []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                # Check for extreme outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                extreme_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                if len(extreme_outliers) > 0:
                    validity_issues["extreme_outliers"].append({
                        "column": col,
                        "count": len(extreme_outliers),
                        "percentage": len(extreme_outliers) / len(col_data) * 100
                    })
        
        return validity_issues
    
    async def _analyze_correlations_with_data(self, data: pd.DataFrame, 
                                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive correlation analysis with actual data"""
        try:
            correlation_results = {
                "numeric_correlations": self._analyze_numeric_correlations(data),
                "categorical_associations": self._analyze_categorical_associations(data),
                "mixed_correlations": self._analyze_mixed_correlations(data),
                "correlation_insights": self._extract_correlation_insights(data),
                "feature_relationships": self._identify_feature_relationships(data)
            }
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {"error": str(e), "analysis_type": "correlation_failed"}

    def _perform_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Basic correlation analysis for comprehensive analysis"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}
        
        # Pearson correlation
        pearson_corr = numeric_data.corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                pearson_val = pearson_corr.iloc[i, j]
                spearman_val = spearman_corr.iloc[i, j]
                
                if abs(pearson_val) > 0.7 or abs(spearman_val) > 0.7:
                    strong_correlations.append({
                        "variable_1": col1,
                        "variable_2": col2,
                        "pearson_correlation": float(pearson_val),
                        "spearman_correlation": float(spearman_val),
                        "strength": "strong" if max(abs(pearson_val), abs(spearman_val)) > 0.8 else "moderate"
                    })
        
        return {
            "pearson_matrix": pearson_corr.to_dict(),
            "spearman_matrix": spearman_corr.to_dict(),
            "strong_correlations": strong_correlations,
            "correlation_summary": {
                "total_pairs": len(strong_correlations),
                "strongest_correlation": max(strong_correlations, 
                                           key=lambda x: max(abs(x["pearson_correlation"]), 
                                                            abs(x["spearman_correlation"]))) if strong_correlations else None
            }
        }

    def _analyze_numeric_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detailed numeric correlation analysis"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}
        
        correlations = {}
        
        # Calculate different correlation types
        correlations["pearson"] = numeric_data.corr(method='pearson').to_dict()
        correlations["spearman"] = numeric_data.corr(method='spearman').to_dict()
        correlations["kendall"] = numeric_data.corr(method='kendall').to_dict()
        
        # Statistical significance testing
        significance_results = []
        for i, col1 in enumerate(numeric_data.columns):
            for j, col2 in enumerate(numeric_data.columns):
                if i < j:  # Avoid duplicates
                    data1 = numeric_data[col1].dropna()
                    data2 = numeric_data[col2].dropna()
                    
                    # Align the data
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) > 3:
                        aligned_data1 = data1[common_idx]
                        aligned_data2 = data2[common_idx]
                        
                        # Pearson correlation with p-value
                        pearson_r, pearson_p = pearsonr(aligned_data1, aligned_data2)
                        
                        # Spearman correlation with p-value
                        spearman_r, spearman_p = spearmanr(aligned_data1, aligned_data2)
                        
                        significance_results.append({
                            "variable_1": col1,
                            "variable_2": col2,
                            "pearson_r": float(pearson_r),
                            "pearson_p_value": float(pearson_p),
                            "spearman_r": float(spearman_r),
                            "spearman_p_value": float(spearman_p),
                            "sample_size": len(aligned_data1),
                            "significant": pearson_p < 0.05 or spearman_p < 0.05
                        })
        
        correlations["significance_tests"] = significance_results
        
        return correlations

    def _analyze_categorical_associations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze associations between categorical variables"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) < 2:
            return {"message": "Insufficient categorical columns for association analysis"}
        
        associations = []
        
        for i, col1 in enumerate(categorical_cols):
            for j, col2 in enumerate(categorical_cols):
                if i < j:  # Avoid duplicates
                    try:
                        # Create contingency table
                        contingency_table = pd.crosstab(data[col1], data[col2])
                        
                        # Chi-square test
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Cramér's V (effect size)
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        associations.append({
                            "variable_1": col1,
                            "variable_2": col2,
                            "chi_square": float(chi2),
                            "p_value": float(p_value),
                            "degrees_of_freedom": int(dof),
                            "cramers_v": float(cramers_v),
                            "significant": p_value < 0.05,
                            "association_strength": self._interpret_cramers_v(cramers_v)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not analyze association between {col1} and {col2}: {e}")
        
        return {
            "categorical_associations": associations,
            "summary": {
                "total_pairs_tested": len(associations),
                "significant_associations": sum(1 for a in associations if a["significant"])
            }
        }

    def _analyze_mixed_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric and categorical variables"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_cols or not categorical_cols:
            return {"message": "Need both numeric and categorical columns for mixed correlation analysis"}
        
        mixed_correlations = []
        
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                try:
                    # Use mutual information for mixed correlations
                    # Encode categorical variable
                    le = LabelEncoder()
                    cat_encoded = le.fit_transform(data[cat_col].fillna('missing'))
                    num_data = data[num_col].fillna(data[num_col].median())
                    
                    # Calculate mutual information
                    mi_score = mutual_info_regression(cat_encoded.reshape(-1, 1), num_data)[0]
                    
                    # ANOVA F-test for numeric vs categorical
                    groups = [group[num_col].dropna() for name, group in data.groupby(cat_col)]
                    if len(groups) > 1 and all(len(group) > 0 for group in groups):
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        mixed_correlations.append({
                            "numeric_variable": num_col,
                            "categorical_variable": cat_col,
                            "mutual_information": float(mi_score),
                            "anova_f_statistic": float(f_stat),
                            "anova_p_value": float(p_value),
                            "significant": p_value < 0.05
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not analyze mixed correlation between {num_col} and {cat_col}: {e}")
        
        return {
            "mixed_correlations": mixed_correlations,
            "summary": {
                "total_pairs_tested": len(mixed_correlations),
                "significant_relationships": sum(1 for mc in mixed_correlations if mc["significant"])
            }
        }

    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V effect size"""
        if cramers_v < 0.1:
            return "negligible"
        elif cramers_v < 0.3:
            return "small"
        elif cramers_v < 0.5:
            return "medium"
        else:
            return "large"
    
    async def _detect_patterns_with_data(self, data: pd.DataFrame, 
                                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive pattern detection in data"""
        try:
            pattern_results = {
                "clustering_analysis": self._perform_clustering_analysis(data),
                "trend_analysis": self._analyze_trends(data),
                "seasonal_patterns": self._detect_seasonal_patterns(data),
                "anomaly_patterns": self._detect_anomaly_patterns(data),
                "distribution_patterns": self._analyze_distribution_patterns(data)
            }
            
            return pattern_results
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return {"error": str(e), "analysis_type": "pattern_detection_failed"}

    def _perform_clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis to detect patterns"""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return {"message": "Insufficient numeric data for clustering analysis"}
        
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Determine optimal number of clusters using elbow method
            inertias = []
            k_range = range(2, min(11, len(numeric_data) // 2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point (simplified)
            optimal_k = k_range[0]
            if len(inertias) > 2:
                # Simple elbow detection
                diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                optimal_k = k_range[diffs.index(max(diffs))]
            
            # Perform clustering with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Analyze clusters
            cluster_analysis = []
            for cluster_id in range(optimal_k):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = numeric_data[cluster_mask]
                
                cluster_analysis.append({
                    "cluster_id": int(cluster_id),
                    "size": int(cluster_mask.sum()),
                    "percentage": float(cluster_mask.sum() / len(numeric_data) * 100),
                    "centroid": cluster_data.mean().to_dict(),
                    "characteristics": self._describe_cluster_characteristics(cluster_data, numeric_data)
                })
            
            return {
                "optimal_clusters": int(optimal_k),
                "cluster_analysis": cluster_analysis,
                "silhouette_score": self._calculate_silhouette_score(scaled_data, cluster_labels),
                "clustering_summary": {
                    "total_points": len(numeric_data),
                    "features_used": numeric_data.columns.tolist(),
                    "largest_cluster": max(cluster_analysis, key=lambda x: x["size"])["cluster_id"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return {"error": str(e)}

    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in numeric data"""
        numeric_data = data.select_dtypes(include=[np.number])
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        trends = {}
        
        # If we have datetime columns, analyze time-based trends
        if datetime_cols and not numeric_data.empty:
            for date_col in datetime_cols[:1]:  # Analyze first datetime column
                for num_col in numeric_data.columns[:5]:  # Limit to first 5 numeric columns
                    try:
                        # Sort by date and calculate trend
                        sorted_data = data.sort_values(date_col)
                        x_values = np.arange(len(sorted_data))
                        y_values = sorted_data[num_col].dropna()
                        
                        if len(y_values) > 3:
                            # Linear regression for trend
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                x_values[:len(y_values)], y_values
                            )
                            
                            trends[f"{num_col}_vs_{date_col}"] = {
                                "slope": float(slope),
                                "r_squared": float(r_value ** 2),
                                "p_value": float(p_value),
                                "trend_direction": "increasing" if slope > 0 else "decreasing",
                                "trend_strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak",
                                "significant": p_value < 0.05
                            }
                    except Exception as e:
                        logger.warning(f"Could not analyze trend for {num_col}: {e}")
        
        # Analyze general trends in numeric data (without time component)
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 10:
                # Calculate moving averages and trends
                window_size = min(10, len(col_data) // 4)
                if window_size > 1:
                    moving_avg = col_data.rolling(window=window_size).mean()
                    trend_slope = (moving_avg.iloc[-1] - moving_avg.iloc[window_size-1]) / (len(moving_avg) - window_size)
                    
                    trends[f"{col}_general_trend"] = {
                        "moving_average_trend": float(trend_slope),
                        "recent_direction": "increasing" if trend_slope > 0 else "decreasing",
                        "volatility": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
                    }
        
        return trends if trends else {"message": "No clear trends detected in the data"}

    async def _generate_comprehensive_insights(self, data: pd.DataFrame, 
                                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive automated insights from data"""
        try:
            insights = {
                "data_summary_insights": self._generate_summary_insights(data),
                "statistical_insights": self._generate_statistical_insights(data),
                "quality_insights": self._generate_quality_insights(data),
                "relationship_insights": self._generate_relationship_insights(data),
                "business_insights": self._generate_business_insights(data),
                "actionable_recommendations": self._generate_actionable_recommendations(data)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e), "analysis_type": "insights_generation_failed"}

    def _extract_automated_insights(self, data: pd.DataFrame) -> List[str]:
        """Extract key automated insights from data analysis"""
        insights = []
        
        # Data shape insights
        insights.append(f"Dataset contains {len(data):,} rows and {len(data.columns)} columns")
        
        # Missing data insights
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_pct > 10:
            insights.append(f"High missing data rate: {missing_pct:.1f}% of values are missing")
        elif missing_pct > 0:
            insights.append(f"Low missing data rate: {missing_pct:.1f}% of values are missing")
        else:
            insights.append("Complete dataset with no missing values")
        
        # Numeric data insights
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numeric columns for statistical analysis")
            
            # Check for highly skewed data
            for col in numeric_cols[:3]:  # Check first 3 numeric columns
                skewness = stats.skew(data[col].dropna())
                if abs(skewness) > 2:
                    insights.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
        
        # Categorical data insights
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Found {len(categorical_cols)} categorical columns")
            
            # Check for high cardinality
            for col in categorical_cols:
                unique_count = data[col].nunique()
                if unique_count > len(data) * 0.8:
                    insights.append(f"Column '{col}' has very high cardinality ({unique_count} unique values)")
        
        # Duplicate insights
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            insights.append(f"Found {duplicate_count} duplicate rows ({duplicate_count/len(data)*100:.1f}%)")
        
        return insights
    
    def _provide_general_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general data science guidance"""
        return {
            "analysis_approach": [
                "Understand the business problem",
                "Explore and clean the data",
                "Perform statistical analysis",
                "Generate actionable insights",
                "Communicate findings clearly"
            ],
            "tools_and_techniques": {
                "descriptive_statistics": "Mean, median, mode, standard deviation",
                "visualization": "Charts, graphs, and interactive dashboards",
                "hypothesis_testing": "Statistical significance testing",
                "pattern_recognition": "Clustering and classification"
            },
            "deliverables": [
                "Executive summary of findings",
                "Detailed statistical analysis",
                "Interactive visualizations",
                "Actionable recommendations"
            ]
        }
    
    def _classify_analysis_type(self, query: str) -> str:
        """Classify the type of analysis request"""
        if "quality" in query:
            return "data_quality_assessment"
        elif "correlation" in query:
            return "correlation_analysis"
        elif "insights" in query:
            return "insight_generation"
        elif "analyze" in query:
            return "exploratory_analysis"
        else:
            return "general_analysis"

    # Additional helper methods for comprehensive analysis
    
    def _detect_outliers_with_data(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive outlier detection"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {"message": "No numeric columns found for outlier detection"}
        
        outlier_results = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 10:
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                # Z-score method
                z_scores = np.abs(stats.zscore(col_data))
                z_outliers = col_data[z_scores > 3]
                
                outlier_results[col] = {
                    "iqr_outliers": {
                        "count": len(iqr_outliers),
                        "percentage": len(iqr_outliers) / len(col_data) * 100,
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                    },
                    "z_score_outliers": {
                        "count": len(z_outliers),
                        "percentage": len(z_outliers) / len(col_data) * 100
                    },
                    "extreme_values": {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "range": float(col_data.max() - col_data.min())
                    }
                }
        
        return outlier_results

    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Basic outlier detection for comprehensive analysis"""
        return self._detect_outliers_with_data(data, {})

    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {"message": "No numeric columns found for distribution analysis"}
        
        distribution_analysis = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 10:
                # Basic distribution statistics
                distribution_analysis[col] = {
                    "distribution_type": self._identify_distribution_type(col_data),
                    "normality_test": self._test_normality(col_data),
                    "skewness": float(stats.skew(col_data)),
                    "kurtosis": float(stats.kurtosis(col_data)),
                    "distribution_summary": {
                        "symmetric": abs(stats.skew(col_data)) < 0.5,
                        "normal_like": self._test_normality(col_data)["is_normal"],
                        "heavy_tailed": stats.kurtosis(col_data) > 3
                    }
                }
        
        return distribution_analysis

    def _identify_distribution_type(self, data: pd.Series) -> str:
        """Identify the likely distribution type"""
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "approximately_normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "unknown"

    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test"""
        if len(data) < 3:
            return {"is_normal": False, "reason": "insufficient_data"}
        
        # Use Shapiro-Wilk for small samples, Anderson-Darling for larger
        if len(data) <= 5000:
            try:
                statistic, p_value = stats.shapiro(data)
                return {
                    "test": "shapiro_wilk",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
            except:
                return {"is_normal": False, "reason": "test_failed"}
        else:
            # For large samples, use a simpler approach
            skewness = abs(stats.skew(data))
            kurtosis = abs(stats.kurtosis(data))
            return {
                "test": "skewness_kurtosis",
                "is_normal": skewness < 2 and kurtosis < 7,
                "skewness": float(skewness),
                "kurtosis": float(kurtosis)
            }

    def _detect_basic_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect basic patterns in data"""
        patterns = {
            "data_patterns": [],
            "column_patterns": {},
            "relationship_patterns": []
        }
        
        # Check for common data patterns
        if len(data.columns) > len(data):
            patterns["data_patterns"].append("Wide dataset (more columns than rows)")
        elif len(data) > len(data.columns) * 100:
            patterns["data_patterns"].append("Long dataset (many more rows than columns)")
        
        # Check column patterns
        for col in data.columns:
            col_patterns = []
            
            # Check for constant values
            if data[col].nunique() == 1:
                col_patterns.append("constant_values")
            
            # Check for sequential patterns (if numeric)
            if data[col].dtype in ['int64', 'float64']:
                if data[col].is_monotonic_increasing:
                    col_patterns.append("monotonic_increasing")
                elif data[col].is_monotonic_decreasing:
                    col_patterns.append("monotonic_decreasing")
            
            if col_patterns:
                patterns["column_patterns"][col] = col_patterns
        
        return patterns

    def _generate_visualization_recommendations(self, data: Optional[pd.DataFrame], 
                                              analysis_type: str, 
                                              analysis_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate visualization recommendations based on data characteristics"""
        if data is None:
            return [
                {"type": "info", "message": "Upload data to get specific visualization recommendations"},
                {"type": "general", "charts": ["histogram", "scatter_plot", "correlation_heatmap", "box_plot"]}
            ]
        
        recommendations = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Univariate visualizations
        if numeric_cols:
            recommendations.append({
                "type": "univariate_numeric",
                "charts": ["histogram", "box_plot", "violin_plot"],
                "columns": numeric_cols[:5],  # Limit recommendations
                "purpose": "Understand distribution and detect outliers"
            })
        
        if categorical_cols:
            recommendations.append({
                "type": "univariate_categorical",
                "charts": ["bar_chart", "pie_chart"],
                "columns": categorical_cols[:3],
                "purpose": "Understand category frequencies"
            })
        
        # Bivariate visualizations
        if len(numeric_cols) >= 2:
            recommendations.append({
                "type": "bivariate_numeric",
                "charts": ["scatter_plot", "correlation_heatmap"],
                "columns": numeric_cols[:4],
                "purpose": "Explore relationships between numeric variables"
            })
        
        if numeric_cols and categorical_cols:
            recommendations.append({
                "type": "bivariate_mixed",
                "charts": ["grouped_bar_chart", "box_plot_by_category"],
                "numeric_columns": numeric_cols[:2],
                "categorical_columns": categorical_cols[:2],
                "purpose": "Compare numeric values across categories"
            })
        
        # Time series visualizations
        if datetime_cols and numeric_cols:
            recommendations.append({
                "type": "time_series",
                "charts": ["line_plot", "area_chart"],
                "datetime_columns": datetime_cols[:1],
                "numeric_columns": numeric_cols[:3],
                "purpose": "Analyze trends over time"
            })
        
        # Analysis-specific recommendations
        if analysis_type == "correlation_analysis":
            recommendations.append({
                "type": "correlation_specific",
                "charts": ["correlation_heatmap", "scatter_matrix"],
                "purpose": "Visualize correlation patterns"
            })
        elif analysis_type == "quality_assessment":
            recommendations.append({
                "type": "quality_specific",
                "charts": ["missing_value_heatmap", "outlier_detection_plot"],
                "purpose": "Visualize data quality issues"
            })
        
        return recommendations

    def _calculate_confidence_score(self, data: Optional[pd.DataFrame], 
                                   result: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for the analysis"""
        if data is None:
            return 0.3  # Low confidence without data
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on data quality
        if len(data) > 100:
            confidence += 0.2
        if len(data) > 1000:
            confidence += 0.1
        
        # Adjust based on missing data
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_pct < 5:
            confidence += 0.1
        elif missing_pct > 20:
            confidence -= 0.2
        
        # Adjust based on data variety
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        if numeric_cols >= 3:
            confidence += 0.1
        
        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0

    def _generate_suggestions(self, query: str, data: Optional[pd.DataFrame], 
                            result: Optional[Dict[str, Any]]) -> List[str]:
        """Generate helpful suggestions based on the analysis"""
        suggestions = []
        
        if data is None:
            suggestions.extend([
                "Upload a dataset to perform actual data analysis",
                "Supported formats: CSV, Excel, JSON",
                "Try asking: 'Analyze my sales data' after uploading"
            ])
        else:
            # Data-specific suggestions
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) >= 2:
                suggestions.append("Try correlation analysis to find relationships between variables")
            
            if len(categorical_cols) > 0:
                suggestions.append("Explore categorical associations with chi-square tests")
            
            if len(data) > 1000:
                suggestions.append("Consider clustering analysis to identify data patterns")
            
            # Missing data suggestions
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            if missing_pct > 10:
                suggestions.append("Address missing data issues before advanced analysis")
            
            # Analysis-specific suggestions
            if "quality" not in query.lower():
                suggestions.append("Run data quality assessment to identify potential issues")
            
            if "visualization" not in query.lower():
                suggestions.append("Create visualizations to better understand your data patterns")
        
        return suggestions

    # Guidance methods for when no data is provided
    
    def _provide_quality_guidance(self) -> Dict[str, Any]:
        """Provide data quality assessment guidance"""
        return {
            "data_quality_framework": {
                "completeness": "Measure missing values and null percentages",
                "consistency": "Check data formats, value ranges, and business rules",
                "accuracy": "Detect outliers, anomalies, and invalid values",
                "uniqueness": "Identify duplicate records and redundant data",
                "validity": "Ensure data conforms to expected formats and constraints",
                "timeliness": "Check if data is current and up-to-date"
            },
            "quality_metrics": [
                "Missing value percentage by column",
                "Duplicate record count and percentage",
                "Outlier detection using IQR and Z-score methods",
                "Data type consistency checks",
                "Value range validation",
                "Format consistency analysis"
            ],
            "recommendations": [
                "Upload your dataset for automated quality assessment",
                "Define business rules for validation",
                "Set acceptable thresholds for missing data",
                "Implement data cleaning procedures"
            ],
            "next_steps": [
                "Upload data file (CSV, Excel, or JSON)",
                "Run comprehensive quality assessment",
                "Review quality metrics and recommendations",
                "Implement data cleaning strategies"
            ]
        }

    def _provide_correlation_guidance(self) -> Dict[str, Any]:
        """Provide correlation analysis guidance"""
        return {
            "correlation_types": {
                "pearson": "Measures linear relationships between numeric variables",
                "spearman": "Measures monotonic relationships (rank-based)",
                "kendall": "Alternative rank-based correlation measure",
                "chi_square": "Tests associations between categorical variables",
                "mutual_information": "Measures dependencies between mixed variable types"
            },
            "interpretation": {
                "strong_correlation": "Absolute value > 0.7",
                "moderate_correlation": "Absolute value 0.3 - 0.7",
                "weak_correlation": "Absolute value < 0.3",
                "statistical_significance": "P-value < 0.05 indicates significant relationship"
            },
            "use_cases": [
                "Feature selection for machine learning",
                "Understanding business relationships",
                "Identifying redundant variables",
                "Detecting multicollinearity issues"
            ],
            "recommendations": [
                "Upload dataset with multiple numeric variables",
                "Include both numeric and categorical variables for comprehensive analysis",
                "Consider domain knowledge when interpreting correlations",
                "Remember: correlation does not imply causation"
            ]
        }

    def _provide_pattern_guidance(self) -> Dict[str, Any]:
        """Provide pattern detection guidance"""
        return {
            "pattern_types": {
                "clustering": "Group similar data points together",
                "trends": "Identify increasing/decreasing patterns over time",
                "seasonality": "Detect recurring patterns in time series",
                "anomalies": "Find unusual or unexpected data points",
                "distributions": "Understand how data values are spread"
            },
            "techniques": {
                "k_means_clustering": "Partition data into k clusters",
                "hierarchical_clustering": "Build cluster hierarchy",
                "time_series_decomposition": "Separate trend, seasonal, and residual components",
                "outlier_detection": "Statistical and machine learning approaches",
                "distribution_fitting": "Test for normal, exponential, and other distributions"
            },
            "applications": [
                "Customer segmentation",
                "Market basket analysis",
                "Fraud detection",
                "Quality control",
                "Predictive maintenance"
            ],
            "requirements": [
                "Sufficient data points (>100 recommended)",
                "Multiple numeric variables for clustering",
                "Time-based data for trend analysis",
                "Clean data with minimal missing values"
            ]
        }

    def _provide_insights_guidance(self) -> Dict[str, Any]:
        """Provide insights generation guidance"""
        return {
            "insight_categories": {
                "descriptive": "What happened in the data?",
                "diagnostic": "Why did it happen?",
                "predictive": "What might happen next?",
                "prescriptive": "What should we do about it?"
            },
            "automated_insights": [
                "Statistical summaries and key metrics",
                "Data quality assessment results",
                "Correlation and relationship findings",
                "Pattern and anomaly detection",
                "Distribution characteristics",
                "Trend analysis and forecasting opportunities"
            ],
            "business_value": [
                "Data-driven decision making",
                "Risk identification and mitigation",
                "Opportunity discovery",
                "Process optimization",
                "Performance monitoring"
            ],
            "best_practices": [
                "Combine automated insights with domain expertise",
                "Validate findings with additional data",
                "Consider business context and constraints",
                "Communicate insights clearly to stakeholders",
                "Track insight accuracy and business impact"
            ]
        }

    def _provide_general_guidance(self, query: str) -> Dict[str, Any]:
        """Provide general data science guidance"""
        return {
            "data_science_process": {
                "1_understand_problem": "Define business questions and objectives",
                "2_collect_data": "Gather relevant, high-quality data",
                "3_explore_data": "Perform exploratory data analysis (EDA)",
                "4_prepare_data": "Clean, transform, and feature engineer",
                "5_analyze_data": "Apply statistical and ML techniques",
                "6_interpret_results": "Extract insights and recommendations",
                "7_communicate_findings": "Present results to stakeholders"
            },
            "available_analyses": {
                "exploratory_analysis": "Comprehensive data exploration and profiling",
                "quality_assessment": "Data quality metrics and recommendations",
                "correlation_analysis": "Relationship analysis between variables",
                "pattern_detection": "Clustering, trends, and anomaly detection",
                "statistical_testing": "Hypothesis testing and significance analysis",
                "visualization_recommendations": "Optimal charts for your data"
            },
            "getting_started": [
                "Upload your dataset (CSV, Excel, or JSON format)",
                "Start with exploratory analysis to understand your data",
                "Check data quality before advanced analysis",
                "Ask specific questions about your data",
                "Request visualizations to better understand patterns"
            ],
            "example_queries": [
                "Analyze my sales data",
                "Check the quality of my customer dataset",
                "Find correlations in my marketing data",
                "Detect patterns in my transaction data",
                "Generate insights from my survey responses"
            ]
        }

    # Additional helper methods
    
    def _calculate_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        score = 100.0
        
        # Penalize for missing data
        missing_pct = quality_metrics["completeness"]["missing_percentage"]
        score -= missing_pct * 0.5  # 0.5 points per percent missing
        
        # Penalize for duplicates
        duplicate_pct = quality_metrics["uniqueness"]["duplicate_percentage"]
        score -= duplicate_pct * 0.3  # 0.3 points per percent duplicate
        
        # Penalize for consistency issues
        consistency_issues = quality_metrics["consistency"]
        if consistency_issues["mixed_data_types"]:
            score -= len(consistency_issues["mixed_data_types"]) * 5
        if consistency_issues["case_inconsistencies"]:
            score -= len(consistency_issues["case_inconsistencies"]) * 2
        
        return max(0.0, min(100.0, score))

    def _generate_quality_recommendations(self, quality_metrics: Dict[str, Any], 
                                        data: pd.DataFrame) -> List[str]:
        """Generate specific data quality recommendations"""
        recommendations = []
        
        # Missing data recommendations
        missing_pct = quality_metrics["completeness"]["missing_percentage"]
        if missing_pct > 20:
            recommendations.append("High missing data rate - consider data collection improvements")
        elif missing_pct > 5:
            recommendations.append("Moderate missing data - implement imputation strategies")
        
        # Duplicate recommendations
        duplicate_pct = quality_metrics["uniqueness"]["duplicate_percentage"]
        if duplicate_pct > 5:
            recommendations.append("Remove duplicate records to improve data quality")
        
        # Consistency recommendations
        consistency_issues = quality_metrics["consistency"]
        if consistency_issues["mixed_data_types"]:
            recommendations.append("Standardize data types in columns with mixed types")
        if consistency_issues["case_inconsistencies"]:
            recommendations.append("Standardize text case for categorical variables")
        
        return recommendations

    def _identify_priority_issues(self, quality_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify priority data quality issues"""
        issues = []
        
        missing_pct = quality_metrics["completeness"]["missing_percentage"]
        if missing_pct > 15:
            issues.append({
                "type": "missing_data",
                "severity": "high",
                "description": f"High missing data rate: {missing_pct:.1f}%",
                "impact": "May significantly affect analysis results"
            })
        
        duplicate_pct = quality_metrics["uniqueness"]["duplicate_percentage"]
        if duplicate_pct > 10:
            issues.append({
                "type": "duplicates",
                "severity": "medium",
                "description": f"High duplicate rate: {duplicate_pct:.1f}%",
                "impact": "May skew statistical analysis"
            })
        
        return issues

    def _generate_health_summary(self, quality_metrics: Dict[str, Any], 
                                quality_score: float) -> str:
        """Generate a health summary for the dataset"""
        if quality_score >= 90:
            return "Excellent data quality - ready for advanced analysis"
        elif quality_score >= 75:
            return "Good data quality - minor issues to address"
        elif quality_score >= 60:
            return "Fair data quality - several issues need attention"
        else:
            return "Poor data quality - significant cleanup required"

    def _extract_correlation_insights(self, data: pd.DataFrame) -> List[str]:
        """Extract key insights from correlation analysis"""
        insights = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            corr_matrix = numeric_data.corr()
            
            # Find strongest correlations
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if strong_corrs:
                strongest = max(strong_corrs, key=lambda x: abs(x[2]))
                insights.append(f"Strongest correlation: {strongest[0]} and {strongest[1]} (r={strongest[2]:.3f})")
            
            # Count significant correlations
            significant_count = sum(1 for _, _, corr in strong_corrs)
            insights.append(f"Found {significant_count} strong correlations (|r| > 0.7)")
        
        return insights

    def _identify_feature_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify important feature relationships"""
        relationships = {
            "strong_positive": [],
            "strong_negative": [],
            "moderate_relationships": [],
            "independence_candidates": []
        }
        
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            corr_matrix = numeric_data.corr()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if corr_val > 0.7:
                        relationships["strong_positive"].append({"var1": col1, "var2": col2, "correlation": float(corr_val)})
                    elif corr_val < -0.7:
                        relationships["strong_negative"].append({"var1": col1, "var2": col2, "correlation": float(corr_val)})
                    elif 0.3 <= abs(corr_val) <= 0.7:
                        relationships["moderate_relationships"].append({"var1": col1, "var2": col2, "correlation": float(corr_val)})
                    elif abs(corr_val) < 0.1:
                        relationships["independence_candidates"].append({"var1": col1, "var2": col2, "correlation": float(corr_val)})
        
        return relationships

    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame, 
                                        full_data: pd.DataFrame) -> Dict[str, str]:
        """Describe characteristics of a cluster"""
        characteristics = {}
        
        for col in cluster_data.columns:
            cluster_mean = cluster_data[col].mean()
            full_mean = full_data[col].mean()
            
            if cluster_mean > full_mean * 1.2:
                characteristics[col] = "above_average"
            elif cluster_mean < full_mean * 0.8:
                characteristics[col] = "below_average"
            else:
                characteristics[col] = "average"
        
        return characteristics

    def _calculate_silhouette_score(self, scaled_data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(scaled_data, labels))
        except:
            return 0.0

    def _detect_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in time series data"""
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not datetime_cols:
            return {"message": "No datetime columns found for seasonal analysis"}
        
        seasonal_patterns = {}
        
        for date_col in datetime_cols[:1]:  # Analyze first datetime column
            try:
                # Extract time components
                data_copy = data.copy()
                data_copy['month'] = data_copy[date_col].dt.month
                data_copy['day_of_week'] = data_copy[date_col].dt.dayofweek
                data_copy['hour'] = data_copy[date_col].dt.hour if data_copy[date_col].dt.hour.nunique() > 1 else None
                
                numeric_cols = data.select_dtypes(include=[np.number]).columns[:3]  # Limit to 3 columns
                
                for num_col in numeric_cols:
                    patterns = {}
                    
                    # Monthly patterns
                    monthly_avg = data_copy.groupby('month')[num_col].mean()
                    if monthly_avg.std() > monthly_avg.mean() * 0.1:  # Significant variation
                        patterns['monthly'] = {
                            "has_pattern": True,
                            "peak_month": int(monthly_avg.idxmax()),
                            "low_month": int(monthly_avg.idxmin()),
                            "variation_coefficient": float(monthly_avg.std() / monthly_avg.mean())
                        }
                    
                    # Weekly patterns
                    weekly_avg = data_copy.groupby('day_of_week')[num_col].mean()
                    if weekly_avg.std() > weekly_avg.mean() * 0.1:
                        patterns['weekly'] = {
                            "has_pattern": True,
                            "peak_day": int(weekly_avg.idxmax()),
                            "low_day": int(weekly_avg.idxmin()),
                            "variation_coefficient": float(weekly_avg.std() / weekly_avg.mean())
                        }
                    
                    if patterns:
                        seasonal_patterns[f"{num_col}_patterns"] = patterns
                        
            except Exception as e:
                logger.warning(f"Could not analyze seasonal patterns: {e}")
        
        return seasonal_patterns if seasonal_patterns else {"message": "No clear seasonal patterns detected"}

    def _detect_anomaly_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomaly patterns in data"""
        anomaly_results = {}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        for col in numeric_data.columns[:5]:  # Limit to 5 columns
            col_data = numeric_data[col].dropna()
            if len(col_data) > 10:
                # Statistical anomalies
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                # Points beyond 3 standard deviations
                anomalies = col_data[abs(col_data - mean_val) > 3 * std_val]
                
                if len(anomalies) > 0:
                    anomaly_results[col] = {
                        "anomaly_count": len(anomalies),
                        "anomaly_percentage": len(anomalies) / len(col_data) * 100,
                        "anomaly_values": anomalies.tolist()[:10],  # Limit to 10 examples
                        "detection_method": "statistical_outliers"
                    }
        
        return anomaly_results if anomaly_results else {"message": "No significant anomalies detected"}

    def _analyze_distribution_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution patterns in numeric data"""
        return self._analyze_distributions(data)

    def _generate_summary_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate summary insights about the dataset"""
        insights = []
        
        # Basic dataset insights
        insights.append(f"Dataset contains {len(data):,} records across {len(data.columns)} variables")
        
        # Data type insights
        numeric_count = len(data.select_dtypes(include=[np.number]).columns)
        categorical_count = len(data.select_dtypes(include=['object', 'category']).columns)
        datetime_count = len(data.select_dtypes(include=['datetime64']).columns)
        
        insights.append(f"Data types: {numeric_count} numeric, {categorical_count} categorical, {datetime_count} datetime")
        
        # Memory and size insights
        memory_mb = data.memory_usage(deep=True).sum() / 1024**2
        insights.append(f"Dataset size: {memory_mb:.1f} MB in memory")
        
        return insights

    def _generate_statistical_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate statistical insights"""
        insights = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            # Variability insights
            cv_values = []
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 0 and col_data.mean() != 0:
                    cv = col_data.std() / col_data.mean()
                    cv_values.append((col, cv))
            
            if cv_values:
                most_variable = max(cv_values, key=lambda x: x[1])
                least_variable = min(cv_values, key=lambda x: x[1])
                insights.append(f"Most variable: {most_variable[0]} (CV={most_variable[1]:.2f})")
                insights.append(f"Least variable: {least_variable[0]} (CV={least_variable[1]:.2f})")
        
        return insights

    def _generate_quality_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate data quality insights"""
        insights = []
        
        # Missing data insights
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            insights.append(f"Missing data found in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}")
        else:
            insights.append("No missing data detected - complete dataset")
        
        # Duplicate insights
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            insights.append(f"Found {duplicate_count} duplicate records ({duplicate_count/len(data)*100:.1f}%)")
        
        return insights

    def _generate_relationship_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate insights about variable relationships"""
        insights = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            corr_matrix = numeric_data.corr()
            
            # Find strongest correlation
            max_corr = 0
            max_pair = None
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > max_corr:
                        max_corr = corr_val
                        max_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
            
            if max_pair and max_corr > 0.3:
                insights.append(f"Strongest relationship: {max_pair[0]} and {max_pair[1]} (r={max_corr:.3f})")
        
        return insights

    def _generate_business_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate business-relevant insights"""
        insights = []
        
        # Look for business-relevant column names
        business_keywords = ['revenue', 'sales', 'profit', 'cost', 'price', 'customer', 'user', 'conversion']
        business_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in business_keywords)]
        
        if business_cols:
            insights.append(f"Business metrics detected: {', '.join(business_cols[:3])}")
            
            # Analyze business metrics
            for col in business_cols[:2]:  # Limit to 2 columns
                if data[col].dtype in ['int64', 'float64']:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        insights.append(f"{col}: mean={col_data.mean():.2f}, range={col_data.min():.2f} to {col_data.max():.2f}")
        
        return insights

    def _generate_actionable_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_pct > 10:
            recommendations.append("Address missing data before proceeding with advanced analysis")
        
        # Analysis recommendations
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        if numeric_cols >= 3:
            recommendations.append("Consider dimensionality reduction (PCA) for high-dimensional analysis")
        
        if len(data) > 1000:
            recommendations.append("Dataset size suitable for machine learning model development")
        
        # Visualization recommendations
        recommendations.append("Create visualizations to communicate findings effectively")
        
        return recommendations

    async def _profile_data(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data profiling"""
        try:
            profile = {
                "overview": self._get_data_overview(data),
                "statistical_profile": self._get_statistical_summary(data),
                "quality_profile": self._assess_data_quality_metrics(data),
                "column_profiles": self._profile_individual_columns(data),
                "relationship_profile": self._profile_relationships(data)
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error in data profiling: {e}")
            return {"error": str(e), "analysis_type": "profiling_failed"}

    def _profile_individual_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile individual columns in detail"""
        column_profiles = {}
        
        for col in data.columns[:10]:  # Limit to 10 columns
            col_data = data[col]
            
            profile = {
                "data_type": str(col_data.dtype),
                "non_null_count": int(col_data.count()),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": float(col_data.isnull().sum() / len(col_data) * 100),
                "unique_count": int(col_data.nunique()),
                "unique_percentage": float(col_data.nunique() / len(col_data) * 100)
            }
            
            if col_data.dtype in ['int64', 'float64']:
                # Numeric column profile
                col_clean = col_data.dropna()
                if len(col_clean) > 0:
                    profile.update({
                        "min": float(col_clean.min()),
                        "max": float(col_clean.max()),
                        "mean": float(col_clean.mean()),
                        "median": float(col_clean.median()),
                        "std": float(col_clean.std()),
                        "skewness": float(stats.skew(col_clean)),
                        "kurtosis": float(stats.kurtosis(col_clean))
                    })
            else:
                # Categorical column profile
                profile.update({
                    "most_frequent": str(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    "most_frequent_count": int(col_data.value_counts().iloc[0]) if not col_data.empty else 0,
                    "least_frequent": str(col_data.value_counts().index[-1]) if len(col_data.value_counts()) > 0 else None
                })
            
            column_profiles[col] = profile
        
        return column_profiles

    def _profile_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile relationships between variables"""
        relationships = {
            "correlation_summary": {},
            "association_summary": {},
            "dependency_summary": {}
        }
        
        # Numeric correlations summary
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            corr_matrix = numeric_data.corr()
            
            # Count correlations by strength
            strong_count = 0
            moderate_count = 0
            weak_count = 0
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.7:
                        strong_count += 1
                    elif corr_val > 0.3:
                        moderate_count += 1
                    else:
                        weak_count += 1
            
            relationships["correlation_summary"] = {
                "total_pairs": strong_count + moderate_count + weak_count,
                "strong_correlations": strong_count,
                "moderate_correlations": moderate_count,
                "weak_correlations": weak_count
            }
        
        return relationships

    def _classify_analysis_type(self, query: str) -> str:
        """Classify the type of analysis request"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["quality", "assess", "check"]):
            return "data_quality_assessment"
        elif any(word in query_lower for word in ["correlation", "relationship", "association"]):
            return "correlation_analysis"
        elif any(word in query_lower for word in ["pattern", "cluster", "group"]):
            return "pattern_detection"
        elif any(word in query_lower for word in ["outlier", "anomaly", "unusual"]):
            return "outlier_detection"
        elif any(word in query_lower for word in ["insights", "summary", "overview"]):
            return "insights_generation"
        elif any(word in query_lower for word in ["profile", "describe", "characterize"]):
            return "data_profiling"
        elif any(word in query_lower for word in ["analyze", "analysis", "explore"]):
            return "comprehensive_analysis"
        else:
            return "general_analysis"

    def _generate_analysis_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Data size recommendations
        if len(data) < 100:
            recommendations.append("Small dataset - consider collecting more data for robust analysis")
        elif len(data) > 10000:
            recommendations.append("Large dataset - consider sampling for exploratory analysis")
        
        # Missing data recommendations
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_pct > 15:
            recommendations.append("High missing data rate - implement data imputation strategies")
        
        # Variable type recommendations
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_cols >= 5:
            recommendations.append("Multiple numeric variables - consider correlation analysis and PCA")
        
        if categorical_cols >= 3:
            recommendations.append("Multiple categorical variables - analyze associations with chi-square tests")
        
        # Analysis workflow recommendations
        recommendations.extend([
            "Start with data quality assessment before advanced analysis",
            "Create visualizations to understand data distributions",
            "Consider domain expertise when interpreting statistical results"
        ])
        
        return recommendations