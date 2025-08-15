"""
Analysis Components - Focused, single-responsibility analysis modules

This module contains specialized analysis components that follow the
modular architecture pattern with clear interfaces and dependencies.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import statistics
import numpy as np
from datetime import datetime

from ..core.modular_components import (
    BaseComponent, ComponentType, ComponentMetadata, component_registry
)
from ..core.enhanced_specialized_agent import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

@dataclass
class AnalysisRequest:
    """Standardized analysis request format"""
    request_id: str
    analysis_type: str
    data: Any
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class AnalysisResult:
    """Standardized analysis result format"""
    request_id: str
    analysis_type: str
    results: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class DataValidatorComponent(BaseComponent):
    """Component for validating input data before analysis"""
    
    def __init__(self):
        super().__init__(
            component_id="data_validator",
            component_type=ComponentType.VALIDATOR,
            name="Data Validator",
            version="1.0.0"
        )
        self._validation_rules = {}
    
    async def _initialize_impl(self) -> bool:
        """Initialize validation rules"""
        self._validation_rules = {
            "numerical": self._validate_numerical_data,
            "categorical": self._validate_categorical_data,
            "time_series": self._validate_time_series_data,
            "text": self._validate_text_data
        }
        logger.info("Data validator component initialized")
        return True
    
    async def _shutdown_impl(self) -> bool:
        """Cleanup validation resources"""
        self._validation_rules.clear()
        logger.info("Data validator component shutdown")
        return True
    
    async def validate_data(self, data: Any, data_type: str) -> Dict[str, Any]:
        """Validate data based on type"""
        if data_type not in self._validation_rules:
            return {
                "valid": False,
                "error": f"Unknown data type: {data_type}",
                "suggestions": list(self._validation_rules.keys())
            }
        
        try:
            validator = self._validation_rules[data_type]
            return await validator(data)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {str(e)}",
                "suggestions": ["Check data format and try again"]
            }
    
    async def _validate_numerical_data(self, data: Any) -> Dict[str, Any]:
        """Validate numerical data"""
        if not isinstance(data, (list, tuple, np.ndarray)):
            return {"valid": False, "error": "Data must be a list or array"}
        
        try:
            numeric_data = [float(x) for x in data]
            
            if len(numeric_data) == 0:
                return {"valid": False, "error": "Data cannot be empty"}
            
            # Check for invalid values
            invalid_count = sum(1 for x in numeric_data if np.isnan(x) or np.isinf(x))
            
            return {
                "valid": True,
                "data_points": len(numeric_data),
                "invalid_values": invalid_count,
                "data_range": [min(numeric_data), max(numeric_data)] if numeric_data else [0, 0],
                "suggestions": ["Remove invalid values"] if invalid_count > 0 else []
            }
        except (ValueError, TypeError):
            return {"valid": False, "error": "Data contains non-numeric values"}
    
    async def _validate_categorical_data(self, data: Any) -> Dict[str, Any]:
        """Validate categorical data"""
        if not isinstance(data, (list, tuple)):
            return {"valid": False, "error": "Categorical data must be a list"}
        
        if len(data) == 0:
            return {"valid": False, "error": "Data cannot be empty"}
        
        unique_values = list(set(data))
        
        return {
            "valid": True,
            "data_points": len(data),
            "unique_categories": len(unique_values),
            "categories": unique_values[:10],  # Show first 10
            "suggestions": ["Consider encoding categories as numbers"] if len(unique_values) > 50 else []
        }
    
    async def _validate_time_series_data(self, data: Any) -> Dict[str, Any]:
        """Validate time series data"""
        if not isinstance(data, (list, tuple)):
            return {"valid": False, "error": "Time series data must be a list"}
        
        if len(data) < 2:
            return {"valid": False, "error": "Time series needs at least 2 data points"}
        
        # Check if data points have timestamp and value
        try:
            for point in data[:5]:  # Check first 5 points
                if not isinstance(point, dict) or 'timestamp' not in point or 'value' not in point:
                    return {"valid": False, "error": "Each data point must have 'timestamp' and 'value'"}
        except:
            return {"valid": False, "error": "Invalid time series format"}
        
        return {
            "valid": True,
            "data_points": len(data),
            "time_range": [data[0].get('timestamp'), data[-1].get('timestamp')],
            "suggestions": ["Ensure timestamps are in chronological order"]
        }
    
    async def _validate_text_data(self, data: Any) -> Dict[str, Any]:
        """Validate text data"""
        if isinstance(data, str):
            data = [data]
        elif not isinstance(data, (list, tuple)):
            return {"valid": False, "error": "Text data must be string or list of strings"}
        
        if len(data) == 0:
            return {"valid": False, "error": "Data cannot be empty"}
        
        # Check if all items are strings
        non_string_count = sum(1 for item in data if not isinstance(item, str))
        
        total_length = sum(len(str(item)) for item in data)
        avg_length = total_length / len(data) if data else 0
        
        return {
            "valid": non_string_count == 0,
            "data_points": len(data),
            "non_string_items": non_string_count,
            "average_length": avg_length,
            "total_characters": total_length,
            "suggestions": ["Convert non-string items to strings"] if non_string_count > 0 else []
        }

class StatisticalAnalyzerComponent(BaseComponent):
    """Component for statistical analysis operations"""
    
    def __init__(self):
        super().__init__(
            component_id="statistical_analyzer",
            component_type=ComponentType.ANALYZER,
            name="Statistical Analyzer",
            version="1.0.0"
        )
        self.add_dependency("data_validator")
        self._analysis_methods = {}
    
    async def _initialize_impl(self) -> bool:
        """Initialize statistical analysis methods"""
        self._analysis_methods = {
            "descriptive": self._descriptive_analysis,
            "correlation": self._correlation_analysis,
            "distribution": self._distribution_analysis,
            "outlier_detection": self._outlier_detection,
            "trend_analysis": self._trend_analysis
        }
        logger.info("Statistical analyzer component initialized")
        return True
    
    async def _shutdown_impl(self) -> bool:
        """Cleanup analysis resources"""
        self._analysis_methods.clear()
        logger.info("Statistical analyzer component shutdown")
        return True
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform statistical analysis"""
        start_time = datetime.now()
        
        try:
            # Validate data first
            validator = component_registry.get_component("data_validator")
            if validator and validator.status.value == "ready":
                validation = await validator.component.validate_data(
                    request.data, request.parameters.get("data_type", "numerical")
                )
                
                if not validation.get("valid", False):
                    return AnalysisResult(
                        request_id=request.request_id,
                        analysis_type=request.analysis_type,
                        results={"error": validation.get("error", "Validation failed")},
                        confidence=0.0,
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        metadata={"validation": validation}
                    )
            
            # Perform analysis
            if request.analysis_type not in self._analysis_methods:
                available_methods = list(self._analysis_methods.keys())
                return AnalysisResult(
                    request_id=request.request_id,
                    analysis_type=request.analysis_type,
                    results={"error": f"Unknown analysis type: {request.analysis_type}"},
                    confidence=0.0,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"available_methods": available_methods}
                )
            
            method = self._analysis_methods[request.analysis_type]
            results = await method(request.data, request.parameters)
            
            return AnalysisResult(
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                results=results,
                confidence=results.get("confidence", 0.8),
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"method": request.analysis_type}
            )
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return AnalysisResult(
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                results={"error": str(e)},
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"exception": str(e)}
            )
    
    async def _descriptive_analysis(self, data: List[float], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        try:
            numeric_data = [float(x) for x in data if not (np.isnan(float(x)) or np.isinf(float(x)))]
            
            if len(numeric_data) == 0:
                return {"error": "No valid numeric data points"}
            
            results = {
                "count": len(numeric_data),
                "mean": statistics.mean(numeric_data),
                "median": statistics.median(numeric_data),
                "mode": statistics.mode(numeric_data) if len(set(numeric_data)) < len(numeric_data) else None,
                "std_dev": statistics.stdev(numeric_data) if len(numeric_data) > 1 else 0,
                "variance": statistics.variance(numeric_data) if len(numeric_data) > 1 else 0,
                "min": min(numeric_data),
                "max": max(numeric_data),
                "range": max(numeric_data) - min(numeric_data),
                "confidence": 0.9
            }
            
            # Add quartiles if requested
            if parameters.get("include_quartiles", True):
                sorted_data = sorted(numeric_data)
                n = len(sorted_data)
                results.update({
                    "q1": sorted_data[n // 4] if n > 3 else sorted_data[0],
                    "q3": sorted_data[3 * n // 4] if n > 3 else sorted_data[-1],
                })
                results["iqr"] = results["q3"] - results["q1"]
            
            return results
            
        except Exception as e:
            return {"error": f"Descriptive analysis failed: {str(e)}"}
    
    async def _correlation_analysis(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation analysis"""
        try:
            # Expect data to be a list of [x, y] pairs or dict with x, y keys
            if isinstance(data, dict) and "x" in data and "y" in data:
                x_data = [float(x) for x in data["x"]]
                y_data = [float(y) for y in data["y"]]
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
                x_data = [float(point[0]) for point in data]
                y_data = [float(point[1]) for point in data]
            else:
                return {"error": "Correlation analysis requires paired data (x, y)"}
            
            if len(x_data) != len(y_data) or len(x_data) < 2:
                return {"error": "Need at least 2 paired data points"}
            
            # Calculate Pearson correlation
            correlation = statistics.correlation(x_data, y_data) if len(x_data) > 1 else 0
            
            # Interpret correlation strength
            abs_corr = abs(correlation)
            if abs_corr >= 0.8:
                strength = "very strong"
            elif abs_corr >= 0.6:
                strength = "strong"
            elif abs_corr >= 0.4:
                strength = "moderate"
            elif abs_corr >= 0.2:
                strength = "weak"
            else:
                strength = "very weak"
            
            return {
                "correlation_coefficient": correlation,
                "strength": strength,
                "direction": "positive" if correlation > 0 else "negative" if correlation < 0 else "none",
                "sample_size": len(x_data),
                "confidence": 0.85
            }
            
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    async def _distribution_analysis(self, data: List[float], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data distribution"""
        try:
            numeric_data = [float(x) for x in data if not (np.isnan(float(x)) or np.isinf(float(x)))]
            
            if len(numeric_data) < 3:
                return {"error": "Need at least 3 data points for distribution analysis"}
            
            mean = statistics.mean(numeric_data)
            std_dev = statistics.stdev(numeric_data)
            
            # Simple normality test (check if data roughly follows normal distribution)
            # Count data points within 1, 2, 3 standard deviations
            within_1_std = sum(1 for x in numeric_data if abs(x - mean) <= std_dev)
            within_2_std = sum(1 for x in numeric_data if abs(x - mean) <= 2 * std_dev)
            within_3_std = sum(1 for x in numeric_data if abs(x - mean) <= 3 * std_dev)
            
            n = len(numeric_data)
            pct_1_std = within_1_std / n
            pct_2_std = within_2_std / n
            pct_3_std = within_3_std / n
            
            # Expected percentages for normal distribution
            expected_1_std = 0.68
            expected_2_std = 0.95
            expected_3_std = 0.997
            
            # Simple normality score
            normality_score = (
                1 - abs(pct_1_std - expected_1_std) +
                1 - abs(pct_2_std - expected_2_std) +
                1 - abs(pct_3_std - expected_3_std)
            ) / 3
            
            return {
                "sample_size": n,
                "mean": mean,
                "std_dev": std_dev,
                "within_1_std": pct_1_std,
                "within_2_std": pct_2_std,
                "within_3_std": pct_3_std,
                "normality_score": normality_score,
                "likely_normal": normality_score > 0.8,
                "confidence": 0.75
            }
            
        except Exception as e:
            return {"error": f"Distribution analysis failed: {str(e)}"}
    
    async def _outlier_detection(self, data: List[float], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in the data"""
        try:
            numeric_data = [float(x) for x in data if not (np.isnan(float(x)) or np.isinf(float(x)))]
            
            if len(numeric_data) < 4:
                return {"error": "Need at least 4 data points for outlier detection"}
            
            # IQR method for outlier detection
            sorted_data = sorted(numeric_data)
            n = len(sorted_data)
            
            q1 = sorted_data[n // 4]
            q3 = sorted_data[3 * n // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [x for x in numeric_data if x < lower_bound or x > upper_bound]
            outlier_indices = [i for i, x in enumerate(numeric_data) if x < lower_bound or x > upper_bound]
            
            return {
                "total_points": len(numeric_data),
                "outlier_count": len(outliers),
                "outlier_percentage": len(outliers) / len(numeric_data) * 100,
                "outliers": outliers,
                "outlier_indices": outlier_indices,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "iqr": iqr,
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"error": f"Outlier detection failed: {str(e)}"}
    
    async def _trend_analysis(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in time series or sequential data"""
        try:
            # Handle different data formats
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'value' in data[0]:
                    # Time series format
                    values = [float(point['value']) for point in data]
                else:
                    # Simple list of values
                    values = [float(x) for x in data]
            else:
                return {"error": "Invalid data format for trend analysis"}
            
            if len(values) < 3:
                return {"error": "Need at least 3 data points for trend analysis"}
            
            # Simple linear trend calculation
            n = len(values)
            x = list(range(n))
            
            # Calculate slope using least squares
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean
            
            # Determine trend direction and strength
            if abs(slope) < 0.01:
                trend_direction = "stable"
                trend_strength = "none"
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = "strong" if slope > 1 else "moderate" if slope > 0.1 else "weak"
            else:
                trend_direction = "decreasing"
                trend_strength = "strong" if slope < -1 else "moderate" if slope < -0.1 else "weak"
            
            # Calculate R-squared for trend fit quality
            predicted = [slope * i + intercept for i in x]
            ss_res = sum((values[i] - predicted[i]) ** 2 for i in range(n))
            ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                "data_points": n,
                "slope": slope,
                "intercept": intercept,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "r_squared": r_squared,
                "fit_quality": "good" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "poor",
                "confidence": min(0.9, r_squared + 0.1)
            }
            
        except Exception as e:
            return {"error": f"Trend analysis failed: {str(e)}"}

# Register components
async def register_analysis_components():
    """Register all analysis components"""
    
    # Data Validator
    validator = DataValidatorComponent()
    validator_metadata = ComponentMetadata(
        component_id="data_validator",
        component_type=ComponentType.VALIDATOR,
        name="Data Validator",
        description="Validates input data for analysis operations",
        version="1.0.0",
        interface_version="1.0.0",
        provides=["data_validation"],
        tags=["validation", "data", "preprocessing"]
    )
    
    # Statistical Analyzer
    analyzer = StatisticalAnalyzerComponent()
    analyzer_metadata = ComponentMetadata(
        component_id="statistical_analyzer",
        component_type=ComponentType.ANALYZER,
        name="Statistical Analyzer",
        description="Performs statistical analysis operations",
        version="1.0.0",
        interface_version="1.0.0",
        dependencies=["data_validator"],
        provides=["statistical_analysis"],
        requires=["data_validation"],
        tags=["statistics", "analysis", "mathematics"]
    )
    
    # Register components
    component_registry.register_component(validator, validator_metadata)
    component_registry.register_component(analyzer, analyzer_metadata)
    
    logger.info("Analysis components registered successfully")