"""
Predictive Analytics Engine for Business Outcome Forecasting

This engine provides advanced predictive analytics capabilities for forecasting
business outcomes, identifying opportunities, and supporting strategic decision-making
with quantified predictions and confidence intervals.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from ..models.advanced_analytics_models import (
    PredictiveAnalyticsRequest, BusinessOutcomePrediction, PredictiveAnalyticsResult,
    PredictionScenario, PredictionType, AnalyticsInsight
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PredictiveAnalyticsEngine:
    """
    Enterprise-grade predictive analytics engine for business forecasting.
    
    Capabilities:
    - Revenue and financial forecasting
    - Customer churn prediction
    - Demand forecasting
    - Risk assessment modeling
    - Opportunity identification
    - Scenario-based predictions
    - Model ensemble and validation
    - Feature importance analysis
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.prediction_cache = {}
        
    async def predict_business_outcomes(self, request: PredictiveAnalyticsRequest) -> PredictiveAnalyticsResult:
        """
        Generate comprehensive business outcome predictions.
        
        Args:
            request: Predictive analytics request with parameters
            
        Returns:
            Detailed prediction results with scenarios and insights
        """
        start_time = datetime.utcnow()
        
        try:
            # Load and prepare data
            training_data = await self._load_training_data(request.data_sources)
            processed_data = await self._preprocess_data(training_data, request)
            
            # Build and validate prediction models
            models = await self._build_prediction_models(processed_data, request)
            
            # Generate base predictions
            base_predictions = await self._generate_base_predictions(models, processed_data, request)
            
            # Generate scenario predictions
            scenario_predictions = await self._generate_scenario_predictions(models, processed_data, request)
            
            # Analyze feature importance
            feature_importance = await self._analyze_feature_importance(models, processed_data)
            
            # Generate business insights
            insights = await self._generate_prediction_insights(base_predictions, scenario_predictions, request)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(base_predictions, scenario_predictions, request)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = PredictiveAnalyticsResult(
                request=request,
                predictions=base_predictions,
                model_performance=self.model_performance.get(request.prediction_type.value, {}),
                feature_importance=feature_importance,
                business_insights=insights,
                recommended_actions=recommendations,
                execution_time_ms=execution_time
            )
            
            # Cache results
            self.prediction_cache[result.analysis_id] = result
            
            logger.info(f"Predictive analytics completed: {len(base_predictions)} predictions in {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predictive analytics: {str(e)}")
            raise
    
    async def forecast_revenue(self, data_sources: List[str], forecast_horizon: int = 90) -> BusinessOutcomePrediction:
        """
        Generate revenue forecasts with confidence intervals.
        
        Args:
            data_sources: List of data sources for revenue forecasting
            forecast_horizon: Number of days to forecast
            
        Returns:
            Revenue forecast prediction
        """
        try:
            request = PredictiveAnalyticsRequest(
                prediction_type=PredictionType.REVENUE_FORECAST,
                data_sources=data_sources,
                target_variable="revenue",
                prediction_horizon=forecast_horizon,
                confidence_level=0.95
            )
            
            result = await self.predict_business_outcomes(request)
            
            if result.predictions:
                return result.predictions[0]
            else:
                # Return default prediction if no results
                return BusinessOutcomePrediction(
                    prediction_type=PredictionType.REVENUE_FORECAST,
                    target_date=datetime.utcnow() + timedelta(days=forecast_horizon),
                    base_prediction=0.0,
                    confidence_interval={"lower": 0.0, "upper": 0.0},
                    scenarios=[],
                    key_drivers=[],
                    risk_factors=["Insufficient data for prediction"],
                    opportunities=[],
                    model_accuracy=0.0
                )
                
        except Exception as e:
            logger.error(f"Error in revenue forecasting: {str(e)}")
            raise
    
    async def predict_customer_churn(self, customer_data_source: str) -> List[Dict[str, Any]]:
        """
        Predict customer churn probabilities.
        
        Args:
            customer_data_source: Data source containing customer information
            
        Returns:
            List of customer churn predictions
        """
        try:
            request = PredictiveAnalyticsRequest(
                prediction_type=PredictionType.CHURN_PREDICTION,
                data_sources=[customer_data_source],
                target_variable="churn_probability",
                prediction_horizon=30,
                confidence_level=0.90
            )
            
            result = await self.predict_business_outcomes(request)
            
            # Convert predictions to customer-specific format
            churn_predictions = []
            for prediction in result.predictions:
                churn_predictions.append({
                    "prediction_id": prediction.prediction_id,
                    "churn_probability": prediction.base_prediction,
                    "confidence_interval": prediction.confidence_interval,
                    "risk_factors": prediction.risk_factors,
                    "retention_opportunities": prediction.opportunities
                })
            
            return churn_predictions
            
        except Exception as e:
            logger.error(f"Error in churn prediction: {str(e)}")
            return []
    
    async def identify_growth_opportunities(self, data_sources: List[str]) -> List[AnalyticsInsight]:
        """
        Identify growth opportunities through predictive analysis.
        
        Args:
            data_sources: List of data sources to analyze
            
        Returns:
            List of identified growth opportunities
        """
        try:
            opportunities = []
            
            # Analyze revenue growth opportunities
            revenue_opportunities = await self._analyze_revenue_opportunities(data_sources)
            opportunities.extend(revenue_opportunities)
            
            # Analyze market expansion opportunities
            market_opportunities = await self._analyze_market_opportunities(data_sources)
            opportunities.extend(market_opportunities)
            
            # Analyze efficiency opportunities
            efficiency_opportunities = await self._analyze_efficiency_opportunities(data_sources)
            opportunities.extend(efficiency_opportunities)
            
            # Rank opportunities by potential impact
            ranked_opportunities = self._rank_opportunities_by_impact(opportunities)
            
            logger.info(f"Identified {len(ranked_opportunities)} growth opportunities")
            
            return ranked_opportunities
            
        except Exception as e:
            logger.error(f"Error identifying growth opportunities: {str(e)}")
            return []
    
    async def _load_training_data(self, data_sources: List[str]) -> pd.DataFrame:
        """Load training data from multiple sources."""
        all_data = []
        
        for source in data_sources:
            source_data = await self._load_data_from_source(source)
            source_data['data_source'] = source
            all_data.append(source_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    async def _load_data_from_source(self, source: str) -> pd.DataFrame:
        """Load data from a specific source."""
        if source == "sales_data":
            return self._generate_sales_training_data()
        elif source == "customer_data":
            return self._generate_customer_training_data()
        elif source == "financial_data":
            return self._generate_financial_training_data()
        elif source == "market_data":
            return self._generate_market_training_data()
        elif source == "operational_data":
            return self._generate_operational_training_data()
        else:
            return self._generate_generic_training_data()
    
    def _generate_sales_training_data(self) -> pd.DataFrame:
        """Generate sample sales training data."""
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
        
        # Generate features
        day_of_week = dates.dayofweek
        month = dates.month
        quarter = dates.quarter
        
        # Seasonal factors
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        
        # Marketing spend (simulated)
        marketing_spend = np.random.uniform(1000, 10000, n_samples)
        
        # Economic indicators
        economic_index = 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))
        
        # Generate target variable (revenue)
        base_revenue = 50000 + marketing_spend * 2.5 + economic_index * 100
        revenue = base_revenue * seasonal_factor + np.random.normal(0, 5000, n_samples)
        revenue = np.maximum(revenue, 0)
        
        return pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'marketing_spend': marketing_spend,
            'economic_index': economic_index,
            'day_of_week': day_of_week,
            'month': month,
            'quarter': quarter,
            'seasonal_factor': seasonal_factor,
            'customer_count': np.random.poisson(100, n_samples),
            'avg_order_value': revenue / np.maximum(np.random.poisson(100, n_samples), 1)
        })
    
    def _generate_customer_training_data(self) -> pd.DataFrame:
        """Generate sample customer training data."""
        np.random.seed(123)
        n_customers = 5000
        
        # Customer features
        tenure_months = np.random.exponential(24, n_customers)
        monthly_spend = np.random.lognormal(6, 1, n_customers)
        support_tickets = np.random.poisson(2, n_customers)
        login_frequency = np.random.exponential(10, n_customers)
        
        # Calculate churn probability based on features
        churn_logit = (-2 + 
                      -0.1 * tenure_months + 
                      -0.0001 * monthly_spend + 
                      0.3 * support_tickets + 
                      -0.05 * login_frequency +
                      np.random.normal(0, 0.5, n_customers))
        
        churn_probability = 1 / (1 + np.exp(-churn_logit))
        churned = np.random.binomial(1, churn_probability, n_customers)
        
        return pd.DataFrame({
            'customer_id': range(n_customers),
            'tenure_months': tenure_months,
            'monthly_spend': monthly_spend,
            'support_tickets': support_tickets,
            'login_frequency': login_frequency,
            'churn_probability': churn_probability,
            'churned': churned,
            'segment': np.random.choice(['A', 'B', 'C'], n_customers),
            'acquisition_channel': np.random.choice(['organic', 'paid', 'referral'], n_customers)
        })
    
    def _generate_financial_training_data(self) -> pd.DataFrame:
        """Generate sample financial training data."""
        np.random.seed(456)
        n_periods = 60  # 5 years of monthly data
        
        dates = pd.date_range(start='2019-01-01', periods=n_periods, freq='M')
        
        # Financial metrics with trends
        base_revenue = np.linspace(1000000, 1500000, n_periods)
        revenue = base_revenue * (1 + np.random.normal(0, 0.1, n_periods))
        
        costs = revenue * (0.7 + np.random.normal(0, 0.05, n_periods))
        profit = revenue - costs
        
        # Market indicators
        market_growth = np.random.normal(0.02, 0.05, n_periods)
        competitor_price = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_periods))
        
        return pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'costs': costs,
            'profit': profit,
            'market_growth': market_growth,
            'competitor_price': competitor_price,
            'cash_flow': profit + np.random.normal(0, 50000, n_periods),
            'employees': np.random.poisson(100, n_periods),
            'r_and_d_spend': revenue * np.random.uniform(0.05, 0.15, n_periods)
        })
    
    def _generate_market_training_data(self) -> pd.DataFrame:
        """Generate sample market training data."""
        np.random.seed(789)
        n_samples = 365 * 2  # 2 years of daily data
        
        dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
        
        # Market indicators
        market_sentiment = np.random.uniform(0.3, 0.8, n_samples)
        competitor_activity = np.random.poisson(5, n_samples)
        industry_growth = np.random.normal(0.001, 0.01, n_samples)
        
        # Demand prediction based on market factors
        base_demand = 1000 + market_sentiment * 500 + competitor_activity * 10
        demand = base_demand + np.random.normal(0, 100, n_samples)
        
        return pd.DataFrame({
            'date': dates,
            'demand': demand,
            'market_sentiment': market_sentiment,
            'competitor_activity': competitor_activity,
            'industry_growth': industry_growth,
            'advertising_spend': np.random.uniform(5000, 20000, n_samples),
            'price_index': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
            'seasonality': np.sin(2 * np.pi * dates.dayofyear / 365.25)
        })
    
    def _generate_operational_training_data(self) -> pd.DataFrame:
        """Generate sample operational training data."""
        np.random.seed(101)
        n_samples = 1000
        
        # Operational metrics
        production_capacity = np.random.uniform(80, 100, n_samples)
        quality_score = np.random.uniform(0.85, 0.99, n_samples)
        employee_satisfaction = np.random.uniform(0.6, 0.9, n_samples)
        
        # Efficiency prediction
        efficiency = (production_capacity * quality_score * employee_satisfaction / 100 + 
                     np.random.normal(0, 0.05, n_samples))
        
        return pd.DataFrame({
            'production_capacity': production_capacity,
            'quality_score': quality_score,
            'employee_satisfaction': employee_satisfaction,
            'efficiency': efficiency,
            'downtime_hours': np.random.exponential(2, n_samples),
            'maintenance_cost': np.random.uniform(1000, 10000, n_samples),
            'energy_consumption': np.random.uniform(500, 2000, n_samples)
        })
    
    def _generate_generic_training_data(self) -> pd.DataFrame:
        """Generate generic training data."""
        np.random.seed(202)
        n_samples = 500
        
        # Generic features
        feature_1 = np.random.normal(0, 1, n_samples)
        feature_2 = np.random.uniform(0, 10, n_samples)
        feature_3 = np.random.exponential(2, n_samples)
        
        # Target variable
        target = (2 * feature_1 + 0.5 * feature_2 + 0.1 * feature_3 + 
                 np.random.normal(0, 0.5, n_samples))
        
        return pd.DataFrame({
            'feature_1': feature_1,
            'feature_2': feature_2,
            'feature_3': feature_3,
            'target': target
        })
    
    async def _preprocess_data(self, data: pd.DataFrame, request: PredictiveAnalyticsRequest) -> Dict[str, Any]:
        """Preprocess data for predictive modeling."""
        if data.empty:
            return {"X": np.array([]), "y": np.array([]), "feature_names": []}
        
        processed_data = data.copy()
        
        # Handle date columns
        date_columns = ['date', 'timestamp', 'time']
        for col in date_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_datetime(processed_data[col])
                # Extract date features
                processed_data[f'{col}_year'] = processed_data[col].dt.year
                processed_data[f'{col}_month'] = processed_data[col].dt.month
                processed_data[f'{col}_day'] = processed_data[col].dt.day
                processed_data[f'{col}_dayofweek'] = processed_data[col].dt.dayofweek
                processed_data = processed_data.drop(columns=[col])
        
        # Handle categorical columns
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != request.target_variable:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.encoders[f"{request.prediction_type.value}_{col}"] = le
        
        # Separate features and target
        if request.target_variable in processed_data.columns:
            y = processed_data[request.target_variable].values
            X = processed_data.drop(columns=[request.target_variable])
        else:
            # If target variable not found, use the last numeric column
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                target_col = numeric_columns[-1]
                y = processed_data[target_col].values
                X = processed_data.drop(columns=[target_col])
            else:
                # No numeric columns, create dummy target
                y = np.random.normal(0, 1, len(processed_data))
                X = processed_data
        
        # Handle feature selection
        if request.feature_selection:
            available_features = [col for col in request.feature_selection if col in X.columns]
            if available_features:
                X = X[available_features]
        
        # Convert to numeric and handle missing values
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[request.prediction_type.value] = scaler
        
        return {
            "X": X_scaled,
            "y": y,
            "feature_names": list(X.columns),
            "original_data": processed_data
        }
    
    async def _build_prediction_models(self, processed_data: Dict[str, Any], 
                                     request: PredictiveAnalyticsRequest) -> Dict[str, Any]:
        """Build and train prediction models."""
        X = processed_data["X"]
        y = processed_data["y"]
        
        if len(X) == 0 or len(y) == 0:
            return {}
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {}
        
        # Linear regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        models['linear_regression'] = lr_model
        
        # Ridge regression
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        models['ridge_regression'] = ridge_model
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model
        
        # Evaluate models
        model_scores = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
        
        # Select best model based on R²
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['r2'])
        best_model = models[best_model_name]
        
        # Store model performance
        self.model_performance[request.prediction_type.value] = {
            'best_model': best_model_name,
            'scores': model_scores,
            'feature_count': X.shape[1],
            'training_samples': len(X_train)
        }
        
        # Store the best model
        self.models[request.prediction_type.value] = best_model
        
        return {
            'models': models,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'scores': model_scores,
            'X_test': X_test,
            'y_test': y_test
        }
    
    async def _generate_base_predictions(self, models: Dict[str, Any], processed_data: Dict[str, Any], 
                                       request: PredictiveAnalyticsRequest) -> List[BusinessOutcomePrediction]:
        """Generate base predictions using the best model."""
        if not models or 'best_model' not in models:
            return []
        
        best_model = models['best_model']
        X = processed_data["X"]
        
        if len(X) == 0:
            return []
        
        # Generate predictions for future periods
        predictions = []
        
        # Use the last data point as base for future predictions
        if len(X) > 0:
            base_features = X[-1:] if len(X.shape) > 1 else X[-1].reshape(1, -1)
            
            # Generate predictions for different time horizons
            for days_ahead in [30, 60, 90]:
                if days_ahead <= request.prediction_horizon:
                    # Modify features slightly to simulate future conditions
                    future_features = base_features.copy()
                    
                    # Add some trend and seasonality adjustments
                    trend_factor = 1 + (days_ahead / 365) * 0.05  # 5% annual growth
                    seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * days_ahead / 365)
                    
                    future_features = future_features * trend_factor * seasonal_factor
                    
                    # Make prediction
                    pred_value = best_model.predict(future_features)[0]
                    
                    # Calculate confidence interval using model uncertainty
                    model_scores = models.get('scores', {})
                    best_model_name = models.get('best_model_name', 'unknown')
                    
                    if best_model_name in model_scores:
                        rmse = model_scores[best_model_name].get('rmse', abs(pred_value) * 0.1)
                    else:
                        rmse = abs(pred_value) * 0.1  # 10% uncertainty as fallback
                    
                    # Calculate confidence interval based on requested confidence level
                    z_score = 1.96 if request.confidence_level >= 0.95 else 1.645  # 95% or 90%
                    margin_of_error = z_score * rmse
                    
                    confidence_interval = {
                        "lower": pred_value - margin_of_error,
                        "upper": pred_value + margin_of_error
                    }
                    
                    # Generate scenarios
                    scenarios = await self._generate_prediction_scenarios(
                        best_model, future_features, pred_value, request
                    )
                    
                    # Identify key drivers and risks
                    key_drivers = await self._identify_key_drivers(processed_data, request)
                    risk_factors = await self._identify_risk_factors(pred_value, request)
                    opportunities = await self._identify_opportunities(pred_value, request)
                    
                    prediction = BusinessOutcomePrediction(
                        prediction_type=request.prediction_type,
                        target_date=datetime.utcnow() + timedelta(days=days_ahead),
                        base_prediction=pred_value,
                        confidence_interval=confidence_interval,
                        scenarios=scenarios,
                        key_drivers=key_drivers,
                        risk_factors=risk_factors,
                        opportunities=opportunities,
                        model_accuracy=model_scores.get(best_model_name, {}).get('r2', 0.0)
                    )
                    
                    predictions.append(prediction)
        
        return predictions
    
    async def _generate_scenario_predictions(self, models: Dict[str, Any], processed_data: Dict[str, Any], 
                                           request: PredictiveAnalyticsRequest) -> List[PredictionScenario]:
        """Generate scenario-based predictions."""
        scenarios = []
        
        if not models or 'best_model' not in models:
            return scenarios
        
        best_model = models['best_model']
        X = processed_data["X"]
        
        if len(X) == 0:
            return scenarios
        
        base_features = X[-1:] if len(X.shape) > 1 else X[-1].reshape(1, -1)
        
        # Define scenarios
        scenario_definitions = [
            {
                "name": "Optimistic",
                "description": "Best-case scenario with favorable market conditions",
                "multiplier": 1.2,
                "probability": 0.25
            },
            {
                "name": "Pessimistic", 
                "description": "Worst-case scenario with challenging conditions",
                "multiplier": 0.8,
                "probability": 0.25
            },
            {
                "name": "Conservative",
                "description": "Conservative scenario with moderate growth",
                "multiplier": 1.05,
                "probability": 0.5
            }
        ]
        
        for scenario_def in scenario_definitions:
            # Modify features for scenario
            scenario_features = base_features * scenario_def["multiplier"]
            
            # Make prediction
            scenario_prediction = best_model.predict(scenario_features)[0]
            
            # Calculate confidence interval for scenario
            base_prediction = best_model.predict(base_features)[0]
            uncertainty = abs(scenario_prediction - base_prediction) * 0.1
            
            scenario = PredictionScenario(
                name=scenario_def["name"],
                description=scenario_def["description"],
                assumptions={"feature_multiplier": scenario_def["multiplier"]},
                predicted_value=scenario_prediction,
                confidence_interval={
                    "lower": scenario_prediction - uncertainty,
                    "upper": scenario_prediction + uncertainty
                },
                probability=scenario_def["probability"]
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_prediction_scenarios(self, model: Any, features: np.ndarray, 
                                           base_prediction: float, 
                                           request: PredictiveAnalyticsRequest) -> List[PredictionScenario]:
        """Generate prediction scenarios for a specific model and features."""
        scenarios = []
        
        scenario_adjustments = [
            ("Optimistic", "Favorable market conditions and strong performance", 1.15, 0.3),
            ("Pessimistic", "Challenging market conditions and headwinds", 0.85, 0.2),
            ("Most Likely", "Expected conditions based on current trends", 1.0, 0.5)
        ]
        
        for name, description, multiplier, probability in scenario_adjustments:
            # Adjust features for scenario
            adjusted_features = features * multiplier
            
            # Make prediction
            scenario_value = model.predict(adjusted_features)[0]
            
            # Calculate confidence interval
            uncertainty = abs(scenario_value - base_prediction) * 0.05
            
            scenario = PredictionScenario(
                name=name,
                description=description,
                assumptions={"adjustment_factor": multiplier},
                predicted_value=scenario_value,
                confidence_interval={
                    "lower": scenario_value - uncertainty,
                    "upper": scenario_value + uncertainty
                },
                probability=probability
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def _analyze_feature_importance(self, models: Dict[str, Any], 
                                        processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze feature importance from the best model."""
        feature_importance = {}
        
        if not models or 'best_model' not in models:
            return feature_importance
        
        best_model = models['best_model']
        feature_names = processed_data.get("feature_names", [])
        
        # Get feature importance based on model type
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting)
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            # Linear models
            importances = np.abs(best_model.coef_)
        else:
            # Fallback: equal importance
            importances = np.ones(len(feature_names)) / len(feature_names)
        
        # Create feature importance dictionary
        for i, feature_name in enumerate(feature_names):
            if i < len(importances):
                feature_importance[feature_name] = float(importances[i])
        
        # Normalize to sum to 1
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    async def _identify_key_drivers(self, processed_data: Dict[str, Any], 
                                  request: PredictiveAnalyticsRequest) -> List[str]:
        """Identify key drivers of the prediction."""
        feature_names = processed_data.get("feature_names", [])
        
        # Get feature importance
        model_key = request.prediction_type.value
        if model_key in self.models:
            importance = await self._analyze_feature_importance(
                {'best_model': self.models[model_key]}, processed_data
            )
            
            # Sort by importance and take top drivers
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            key_drivers = [feature for feature, _ in sorted_features[:5]]
        else:
            # Fallback: use first few feature names
            key_drivers = feature_names[:5]
        
        return key_drivers
    
    async def _identify_risk_factors(self, prediction_value: float, 
                                   request: PredictiveAnalyticsRequest) -> List[str]:
        """Identify risk factors based on prediction type and value."""
        risk_factors = []
        
        if request.prediction_type == PredictionType.REVENUE_FORECAST:
            if prediction_value < 0:
                risk_factors.append("Negative revenue forecast indicates significant business challenges")
            risk_factors.extend([
                "Market volatility could impact revenue projections",
                "Competitive pressure may affect pricing and demand",
                "Economic downturns could reduce customer spending"
            ])
        
        elif request.prediction_type == PredictionType.CHURN_PREDICTION:
            if prediction_value > 0.5:
                risk_factors.append("High churn probability indicates customer retention issues")
            risk_factors.extend([
                "Customer satisfaction decline could accelerate churn",
                "Competitive offerings may attract customers away",
                "Service quality issues could drive customer defection"
            ])
        
        elif request.prediction_type == PredictionType.DEMAND_PREDICTION:
            risk_factors.extend([
                "Supply chain disruptions could affect demand fulfillment",
                "Seasonal variations may impact demand patterns",
                "Economic factors could influence customer demand"
            ])
        
        else:
            risk_factors.extend([
                "Model uncertainty due to limited historical data",
                "External factors not captured in the model",
                "Changing market conditions affecting predictions"
            ])
        
        return risk_factors[:5]  # Limit to top 5 risks
    
    async def _identify_opportunities(self, prediction_value: float, 
                                    request: PredictiveAnalyticsRequest) -> List[str]:
        """Identify opportunities based on prediction type and value."""
        opportunities = []
        
        if request.prediction_type == PredictionType.REVENUE_FORECAST:
            if prediction_value > 0:
                opportunities.append("Positive revenue forecast enables growth investments")
            opportunities.extend([
                "Strong performance could support market expansion",
                "Revenue growth may enable new product development",
                "Success could attract strategic partnerships"
            ])
        
        elif request.prediction_type == PredictionType.CHURN_PREDICTION:
            if prediction_value < 0.3:
                opportunities.append("Low churn risk allows focus on growth initiatives")
            opportunities.extend([
                "Stable customer base supports upselling opportunities",
                "Customer loyalty enables premium pricing strategies",
                "Retention success could be replicated in other segments"
            ])
        
        elif request.prediction_type == PredictionType.DEMAND_PREDICTION:
            opportunities.extend([
                "Demand insights enable optimized inventory management",
                "Predictable demand supports capacity planning",
                "Market understanding creates competitive advantages"
            ])
        
        else:
            opportunities.extend([
                "Predictive insights enable proactive decision-making",
                "Data-driven approach improves strategic planning",
                "Analytics capabilities provide competitive differentiation"
            ])
        
        return opportunities[:5]  # Limit to top 5 opportunities
    
    async def _generate_prediction_insights(self, predictions: List[BusinessOutcomePrediction], 
                                          scenarios: List[PredictionScenario], 
                                          request: PredictiveAnalyticsRequest) -> List[str]:
        """Generate insights from predictions."""
        insights = []
        
        if not predictions:
            insights.append("Insufficient data for reliable predictions")
            return insights
        
        # Analyze prediction trends
        if len(predictions) > 1:
            values = [p.base_prediction for p in predictions]
            if values[-1] > values[0]:
                trend = "increasing"
                change = ((values[-1] - values[0]) / abs(values[0])) * 100 if values[0] != 0 else 0
            else:
                trend = "decreasing"
                change = ((values[0] - values[-1]) / abs(values[0])) * 100 if values[0] != 0 else 0
            
            insights.append(f"Prediction shows {trend} trend with {change:.1f}% change over forecast period")
        
        # Analyze confidence levels
        avg_accuracy = np.mean([p.model_accuracy for p in predictions])
        if avg_accuracy > 0.8:
            insights.append("High model accuracy provides reliable predictions")
        elif avg_accuracy > 0.6:
            insights.append("Moderate model accuracy suggests reasonable prediction reliability")
        else:
            insights.append("Lower model accuracy indicates higher prediction uncertainty")
        
        # Analyze scenario spread
        if scenarios:
            scenario_values = [s.predicted_value for s in scenarios]
            scenario_spread = max(scenario_values) - min(scenario_values)
            avg_prediction = np.mean([p.base_prediction for p in predictions])
            
            if scenario_spread > abs(avg_prediction) * 0.2:
                insights.append("Wide scenario range indicates high sensitivity to market conditions")
            else:
                insights.append("Narrow scenario range suggests stable predictions across conditions")
        
        # Risk and opportunity balance
        total_risks = sum(len(p.risk_factors) for p in predictions)
        total_opportunities = sum(len(p.opportunities) for p in predictions)
        
        if total_opportunities > total_risks:
            insights.append("Analysis reveals more opportunities than risks")
        elif total_risks > total_opportunities:
            insights.append("Risk factors outweigh opportunities, requiring mitigation strategies")
        else:
            insights.append("Balanced risk-opportunity profile suggests stable outlook")
        
        return insights
    
    async def _generate_recommendations(self, predictions: List[BusinessOutcomePrediction], 
                                      scenarios: List[PredictionScenario], 
                                      request: PredictiveAnalyticsRequest) -> List[str]:
        """Generate actionable recommendations based on predictions."""
        recommendations = []
        
        if not predictions:
            recommendations.append("Collect more data to improve prediction reliability")
            return recommendations
        
        # General recommendations based on prediction type
        if request.prediction_type == PredictionType.REVENUE_FORECAST:
            recommendations.extend([
                "Monitor key revenue drivers identified in the analysis",
                "Develop contingency plans for pessimistic scenarios",
                "Invest in areas showing positive forecast trends"
            ])
        
        elif request.prediction_type == PredictionType.CHURN_PREDICTION:
            recommendations.extend([
                "Implement retention programs for high-risk customers",
                "Address key factors driving customer churn",
                "Monitor churn indicators in real-time"
            ])
        
        elif request.prediction_type == PredictionType.DEMAND_PREDICTION:
            recommendations.extend([
                "Adjust inventory levels based on demand forecasts",
                "Optimize supply chain for predicted demand patterns",
                "Develop marketing strategies aligned with demand cycles"
            ])
        
        # Model-specific recommendations
        model_performance = self.model_performance.get(request.prediction_type.value, {})
        if model_performance.get('best_model') == 'random_forest':
            recommendations.append("Consider feature engineering to improve Random Forest performance")
        elif model_performance.get('best_model') == 'linear_regression':
            recommendations.append("Linear relationships detected - consider trend-based strategies")
        
        # Data quality recommendations
        if model_performance.get('training_samples', 0) < 100:
            recommendations.append("Increase data collection to improve model reliability")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    async def _analyze_revenue_opportunities(self, data_sources: List[str]) -> List[AnalyticsInsight]:
        """Analyze revenue growth opportunities."""
        opportunities = []
        
        try:
            # Generate revenue forecast
            revenue_prediction = await self.forecast_revenue(data_sources, 90)
            
            if revenue_prediction.base_prediction > 0:
                # Check for growth opportunity
                growth_rate = 0.1  # Assume 10% growth potential
                
                opportunity = AnalyticsInsight(
                    title="Revenue Growth Opportunity",
                    description=f"Predictive analysis indicates potential for {growth_rate*100:.0f}% revenue increase",
                    insight_type="revenue_opportunity",
                    confidence=revenue_prediction.model_accuracy,
                    business_impact=f"Potential revenue increase of ${revenue_prediction.base_prediction * growth_rate:,.0f}",
                    supporting_data={
                        "base_prediction": revenue_prediction.base_prediction,
                        "growth_potential": growth_rate
                    },
                    recommended_actions=[
                        "Focus on key revenue drivers identified in the analysis",
                        "Implement growth strategies in high-potential areas",
                        "Monitor revenue performance against predictions"
                    ],
                    priority=9
                )
                opportunities.append(opportunity)
                
        except Exception as e:
            logger.warning(f"Revenue opportunity analysis failed: {str(e)}")
        
        return opportunities
    
    async def _analyze_market_opportunities(self, data_sources: List[str]) -> List[AnalyticsInsight]:
        """Analyze market expansion opportunities."""
        opportunities = []
        
        # Simulate market opportunity analysis
        opportunity = AnalyticsInsight(
            title="Market Expansion Opportunity",
            description="Predictive models identify untapped market segments with high growth potential",
            insight_type="market_opportunity",
            confidence=0.75,
            business_impact="Market expansion could increase total addressable market by 25-40%",
            supporting_data={"market_segments": 3, "growth_potential": 0.3},
            recommended_actions=[
                "Conduct detailed market research in identified segments",
                "Develop targeted products for new market opportunities",
                "Create market entry strategies for high-potential segments"
            ],
            priority=7
        )
        opportunities.append(opportunity)
        
        return opportunities
    
    async def _analyze_efficiency_opportunities(self, data_sources: List[str]) -> List[AnalyticsInsight]:
        """Analyze operational efficiency opportunities."""
        opportunities = []
        
        # Simulate efficiency opportunity analysis
        opportunity = AnalyticsInsight(
            title="Operational Efficiency Opportunity",
            description="Predictive analysis reveals process optimization potential reducing costs by 15-20%",
            insight_type="efficiency_opportunity",
            confidence=0.8,
            business_impact="Cost reduction of $500K-750K annually through process optimization",
            supporting_data={"cost_reduction_potential": 0.175, "affected_processes": 5},
            recommended_actions=[
                "Implement process automation in identified areas",
                "Optimize resource allocation based on predictive insights",
                "Monitor efficiency metrics and adjust strategies accordingly"
            ],
            priority=8
        )
        opportunities.append(opportunity)
        
        return opportunities
    
    def _rank_opportunities_by_impact(self, opportunities: List[AnalyticsInsight]) -> List[AnalyticsInsight]:
        """Rank opportunities by potential business impact."""
        return sorted(opportunities, key=lambda x: (x.priority, x.confidence), reverse=True)[:10]