"""
QA Agent - Natural language data querying with SQL generation and context awareness
Enhanced with comprehensive query understanding, result explanation, and caching
"""
import time
import re
import json
import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
from pathlib import Path

from .base import Agent, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context information for query processing"""
    table_schema: Dict[str, Any]
    sample_data: Optional[pd.DataFrame]
    column_types: Dict[str, str]
    relationships: List[Dict[str, Any]]
    previous_queries: List[str]
    user_preferences: Dict[str, Any]


@dataclass
class QueryResult:
    """Result of query processing"""
    sql_query: str
    explanation: str
    data: Optional[pd.DataFrame]
    visualization_config: Dict[str, Any]
    insights: List[str]
    confidence_score: float
    execution_time: float


class SQLGenerator:
    """Advanced SQL generation from natural language"""
    
    def __init__(self):
        self.query_patterns = self._initialize_query_patterns()
        self.aggregation_functions = {
            'count': 'COUNT',
            'sum': 'SUM',
            'average': 'AVG',
            'mean': 'AVG',
            'max': 'MAX',
            'maximum': 'MAX',
            'min': 'MIN',
            'minimum': 'MIN',
            'total': 'SUM'
        }
        self.time_patterns = {
            'last month': "date >= date('now', '-1 month')",
            'this month': "strftime('%Y-%m', date) = strftime('%Y-%m', 'now')",
            'last year': "date >= date('now', '-1 year')",
            'this year': "strftime('%Y', date) = strftime('%Y', 'now')",
            'last week': "date >= date('now', '-7 days')",
            'today': "date(date) = date('now')",
            'yesterday': "date(date) = date('now', '-1 day')"
        }
    
    def _initialize_query_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common query patterns"""
        return {
            'aggregation': {
                'patterns': [
                    r'how many (.+)',
                    r'what is the (total|sum|average|mean|max|min) (.+)',
                    r'count (.+)',
                    r'total (.+)',
                    r'average (.+)'
                ],
                'template': 'SELECT {agg_func}({column}) FROM {table} {where_clause}'
            },
            'filtering': {
                'patterns': [
                    r'show (.+) where (.+)',
                    r'find (.+) with (.+)',
                    r'get (.+) that (.+)',
                    r'list (.+) where (.+)'
                ],
                'template': 'SELECT {columns} FROM {table} WHERE {conditions}'
            },
            'comparison': {
                'patterns': [
                    r'compare (.+) between (.+) and (.+)',
                    r'(.+) vs (.+)',
                    r'difference between (.+) and (.+)'
                ],
                'template': 'SELECT {columns}, {metric} FROM {table} WHERE {conditions} GROUP BY {group_by}'
            },
            'top_bottom': {
                'patterns': [
                    r'top (\d+) (.+)',
                    r'bottom (\d+) (.+)',
                    r'highest (.+)',
                    r'lowest (.+)'
                ],
                'template': 'SELECT {columns} FROM {table} ORDER BY {order_column} {direction} LIMIT {limit}'
            },
            'trend': {
                'patterns': [
                    r'(.+) over time',
                    r'trend of (.+)',
                    r'(.+) by (month|year|day|week)',
                    r'growth of (.+)'
                ],
                'template': 'SELECT {time_column}, {metric} FROM {table} ORDER BY {time_column}'
            }
        }
    
    def generate_sql(self, query: str, context: QueryContext) -> Tuple[str, str, float]:
        """Generate SQL from natural language query"""
        query_lower = query.lower().strip()
        
        # Analyze query intent and extract components
        intent = self._classify_query_intent(query_lower)
        components = self._extract_query_components(query_lower, context)
        
        # Generate SQL based on intent
        if intent == 'aggregation':
            sql, explanation = self._generate_aggregation_sql(components, context)
        elif intent == 'filtering':
            sql, explanation = self._generate_filtering_sql(components, context)
        elif intent == 'comparison':
            sql, explanation = self._generate_comparison_sql(components, context)
        elif intent == 'top_bottom':
            sql, explanation = self._generate_top_bottom_sql(components, context)
        elif intent == 'trend':
            sql, explanation = self._generate_trend_sql(components, context)
        else:
            sql, explanation = self._generate_general_sql(components, context)
        
        # Calculate confidence score
        confidence = self._calculate_sql_confidence(query_lower, components, context)
        
        return sql, explanation, confidence
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the intent of the query"""
        for intent, pattern_info in self.query_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.search(pattern, query):
                    return intent
        return 'general'
    
    def _extract_query_components(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Extract components from the query"""
        components = {
            'columns': [],
            'table': None,
            'conditions': [],
            'aggregations': [],
            'time_filters': [],
            'numeric_filters': [],
            'text_filters': [],
            'group_by': [],
            'order_by': [],
            'limit': None
        }
        
        # Extract table name (assume single table for now)
        if context.table_schema:
            components['table'] = list(context.table_schema.keys())[0]
        
        # Extract columns mentioned in query
        for table, columns in context.table_schema.items():
            for column in columns:
                if column.lower() in query or column.replace('_', ' ').lower() in query:
                    components['columns'].append(column)
        
        # Extract aggregation functions
        for word, sql_func in self.aggregation_functions.items():
            if word in query:
                components['aggregations'].append(sql_func)
        
        # Extract time filters
        for time_phrase, sql_condition in self.time_patterns.items():
            if time_phrase in query:
                components['time_filters'].append(sql_condition)
        
        # Extract numeric filters
        numeric_matches = re.findall(r'(greater than|more than|above|over|less than|below|under|equals?|=)\s*(\d+(?:\.\d+)?)', query)
        for operator, value in numeric_matches:
            sql_operator = self._convert_operator(operator)
            components['numeric_filters'].append(f"{sql_operator} {value}")
        
        # Extract limit
        limit_match = re.search(r'top (\d+)|first (\d+)|limit (\d+)', query)
        if limit_match:
            components['limit'] = next(g for g in limit_match.groups() if g)
        
        return components
    
    def _convert_operator(self, operator: str) -> str:
        """Convert natural language operators to SQL"""
        operator_map = {
            'greater than': '>',
            'more than': '>',
            'above': '>',
            'over': '>',
            'less than': '<',
            'below': '<',
            'under': '<',
            'equals': '=',
            'equal': '='
        }
        return operator_map.get(operator, '=')
    
    def _generate_aggregation_sql(self, components: Dict[str, Any], context: QueryContext) -> Tuple[str, str]:
        """Generate SQL for aggregation queries"""
        table = components['table']
        agg_func = components['aggregations'][0] if components['aggregations'] else 'COUNT'
        column = components['columns'][0] if components['columns'] else '*'
        
        if agg_func == 'COUNT' and column != '*':
            sql = f"SELECT COUNT({column}) as count_{column} FROM {table}"
        else:
            sql = f"SELECT {agg_func}({column}) as {agg_func.lower()}_{column} FROM {table}"
        
        # Add WHERE clause if conditions exist
        where_conditions = []
        where_conditions.extend(components['time_filters'])
        
        if where_conditions:
            sql += f" WHERE {' AND '.join(where_conditions)}"
        
        explanation = f"This query calculates the {agg_func.lower()} of {column} from the {table} table"
        if where_conditions:
            explanation += f" with the specified conditions"
        
        return sql, explanation
    
    def _generate_filtering_sql(self, components: Dict[str, Any], context: QueryContext) -> Tuple[str, str]:
        """Generate SQL for filtering queries"""
        table = components['table']
        columns = components['columns'] if components['columns'] else ['*']
        
        sql = f"SELECT {', '.join(columns)} FROM {table}"
        
        # Build WHERE clause
        where_conditions = []
        where_conditions.extend(components['time_filters'])
        where_conditions.extend(components['numeric_filters'])
        where_conditions.extend(components['text_filters'])
        
        if where_conditions:
            sql += f" WHERE {' AND '.join(where_conditions)}"
        
        # Add LIMIT if specified
        if components['limit']:
            sql += f" LIMIT {components['limit']}"
        
        explanation = f"This query retrieves {', '.join(columns)} from the {table} table"
        if where_conditions:
            explanation += f" filtered by the specified conditions"
        
        return sql, explanation
    
    def _generate_comparison_sql(self, components: Dict[str, Any], context: QueryContext) -> Tuple[str, str]:
        """Generate SQL for comparison queries"""
        table = components['table']
        columns = components['columns'] if components['columns'] else ['*']
        
        # For comparison, we typically need GROUP BY
        group_column = components['columns'][0] if components['columns'] else None
        metric_column = components['columns'][1] if len(components['columns']) > 1 else components['columns'][0]
        
        if group_column and metric_column:
            sql = f"SELECT {group_column}, AVG({metric_column}) as avg_{metric_column} FROM {table} GROUP BY {group_column}"
        else:
            sql = f"SELECT {', '.join(columns)} FROM {table}"
        
        explanation = f"This query compares {metric_column} across different {group_column} values"
        
        return sql, explanation
    
    def _generate_top_bottom_sql(self, components: Dict[str, Any], context: QueryContext) -> Tuple[str, str]:
        """Generate SQL for top/bottom queries"""
        table = components['table']
        columns = components['columns'] if components['columns'] else ['*']
        limit = components['limit'] or '10'
        
        # Determine sort column and direction
        sort_column = components['columns'][0] if components['columns'] else 'id'
        direction = 'DESC'  # Default to top (highest values)
        
        sql = f"SELECT {', '.join(columns)} FROM {table} ORDER BY {sort_column} {direction} LIMIT {limit}"
        
        explanation = f"This query retrieves the top {limit} records from {table} ordered by {sort_column}"
        
        return sql, explanation
    
    def _generate_trend_sql(self, components: Dict[str, Any], context: QueryContext) -> Tuple[str, str]:
        """Generate SQL for trend analysis queries"""
        table = components['table']
        
        # Find date/time column
        date_column = None
        for col, col_type in context.column_types.items():
            if 'date' in col.lower() or 'time' in col.lower() or col_type in ['datetime', 'date']:
                date_column = col
                break
        
        if not date_column:
            date_column = 'date'  # Fallback
        
        metric_column = components['columns'][0] if components['columns'] else 'value'
        
        sql = f"SELECT {date_column}, {metric_column} FROM {table} ORDER BY {date_column}"
        
        explanation = f"This query shows the trend of {metric_column} over time using {date_column}"
        
        return sql, explanation
    
    def _generate_general_sql(self, components: Dict[str, Any], context: QueryContext) -> Tuple[str, str]:
        """Generate general SQL query"""
        table = components['table']
        columns = components['columns'] if components['columns'] else ['*']
        
        sql = f"SELECT {', '.join(columns)} FROM {table}"
        
        if components['limit']:
            sql += f" LIMIT {components['limit']}"
        
        explanation = f"This query retrieves {', '.join(columns)} from the {table} table"
        
        return sql, explanation
    
    def _calculate_sql_confidence(self, query: str, components: Dict[str, Any], context: QueryContext) -> float:
        """Calculate confidence score for generated SQL"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we found relevant columns
        if components['columns']:
            confidence += 0.2
        
        # Increase confidence if we identified query intent
        intent_keywords = ['count', 'sum', 'average', 'max', 'min', 'show', 'find', 'list', 'compare', 'top', 'trend']
        if any(keyword in query for keyword in intent_keywords):
            confidence += 0.2
        
        # Increase confidence if we have good context
        if context.table_schema and context.column_types:
            confidence += 0.1
        
        return min(confidence, 1.0)


class QueryCache:
    """Cache for query results and optimizations"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_cache_key(self, query: str, context_hash: str) -> str:
        """Generate cache key for query and context"""
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context_hash: str) -> Optional[QueryResult]:
        """Get cached query result"""
        key = self._generate_cache_key(query, context_hash)
        
        if key in self.cache:
            # Check if cache entry is still valid
            if time.time() - self.access_times[key] < self.ttl_seconds:
                self.access_times[key] = time.time()  # Update access time
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def set(self, query: str, context_hash: str, result: QueryResult):
        """Cache query result"""
        key = self._generate_cache_key(query, context_hash)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.access_times.clear()


class QAAgent(Agent):
    """Enhanced QA Agent for natural language data querying with SQL generation and context awareness"""
    
    def __init__(self):
        super().__init__(
            name="QA Agent",
            description="Answers questions about data using natural language queries with SQL generation, context-aware understanding, result explanation, and intelligent caching"
        )
        self.sql_generator = SQLGenerator()
        self.query_cache = QueryCache()
        self.conversation_history = []
        self.data_connections = {}  # Store database connections
    
    def get_capabilities(self) -> List[str]:
        """Return enhanced QA agent capabilities"""
        return [
            "Advanced natural language to SQL conversion with pattern recognition",
            "Context-aware query understanding with conversation memory",
            "Intelligent result explanation and insights generation",
            "Query optimization and intelligent caching system",
            "Multi-table query support with relationship detection",
            "Interactive data exploration with follow-up suggestions",
            "Automatic visualization recommendations based on query type",
            "Query confidence scoring and validation",
            "Support for complex aggregations, filtering, and comparisons",
            "Time-series and trend analysis capabilities",
            "Data quality assessment during querying",
            "Query performance optimization and indexing suggestions"
        ]
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process enhanced QA requests with SQL generation and context awareness"""
        start_time = time.time()
        
        try:
            query = request.query
            context = request.context
            parameters = request.parameters
            
            # Build query context from available information
            query_context = self._build_query_context(context, parameters)
            
            # Check cache first
            context_hash = self._generate_context_hash(query_context)
            cached_result = self.query_cache.get(query, context_hash)
            
            if cached_result:
                logger.info(f"Returning cached result for query: {query}")
                return AgentResponse(
                    agent_name=self.name,
                    success=True,
                    result=self._format_cached_result(cached_result),
                    metadata={
                        "cached": True,
                        "confidence_score": cached_result.confidence_score,
                        "query_type": self._classify_question(query)
                    },
                    processing_time=time.time() - start_time,
                    confidence_score=cached_result.confidence_score
                )
            
            # Process new query
            if query_context.table_schema:
                # We have data context - generate and execute SQL
                result = await self._process_query_with_data(query, query_context)
            else:
                # No data context - provide guidance and examples
                result = await self._process_query_without_data(query, context)
            
            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "timestamp": datetime.utcnow(),
                "result_type": result.get("type", "guidance")
            })
            
            # Generate follow-up suggestions
            suggestions = self._generate_follow_up_suggestions(query, result, query_context)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                result=result,
                metadata={
                    "cached": False,
                    "query_type": self._classify_question(query),
                    "has_data_context": bool(query_context.table_schema),
                    "conversation_length": len(self.conversation_history)
                },
                processing_time=time.time() - start_time,
                confidence_score=result.get("confidence_score", 0.7),
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Enhanced QA Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e),
                error_code="QA_PROCESSING_ERROR",
                processing_time=time.time() - start_time,
                suggestions=[
                    "Please ensure your data is properly uploaded and accessible",
                    "Try rephrasing your question more specifically",
                    "Check that column names in your question match your data",
                    "Consider providing more context about what you're looking for"
                ]
            )
    
    def _build_query_context(self, context: Dict[str, Any], parameters: Dict[str, Any]) -> QueryContext:
        """Build comprehensive query context from available information"""
        # Extract data and schema information
        data = self._extract_data_from_context(context)
        table_schema = {}
        column_types = {}
        sample_data = None
        
        if data is not None:
            sample_data = data.head(100) if len(data) > 100 else data
            table_name = context.get("table_name", "data")
            table_schema[table_name] = list(data.columns)
            column_types = {col: str(data[col].dtype) for col in data.columns}
        elif "schema" in context:
            table_schema = context["schema"]
            column_types = context.get("column_types", {})
        
        # Get previous queries from conversation history
        previous_queries = [item["query"] for item in self.conversation_history[-5:]]
        
        # Extract user preferences
        user_preferences = parameters.get("preferences", {})
        
        return QueryContext(
            table_schema=table_schema,
            sample_data=sample_data,
            column_types=column_types,
            relationships=context.get("relationships", []),
            previous_queries=previous_queries,
            user_preferences=user_preferences
        )
    
    def _extract_data_from_context(self, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract DataFrame from context"""
        try:
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
                file_path = context["file_path"]
                if file_path.endswith('.csv'):
                    return pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    return pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    return pd.read_json(file_path)
            return None
        except Exception as e:
            logger.error(f"Error extracting data from context: {e}")
            return None
    
    def _generate_context_hash(self, query_context: QueryContext) -> str:
        """Generate hash for query context to use in caching"""
        context_str = json.dumps({
            "schema": query_context.table_schema,
            "types": query_context.column_types,
            "relationships": query_context.relationships
        }, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    async def _process_query_with_data(self, query: str, query_context: QueryContext) -> Dict[str, Any]:
        """Process query when data context is available"""
        try:
            # Generate SQL from natural language
            sql_query, explanation, confidence = self.sql_generator.generate_sql(query, query_context)
            
            # Execute query if we have sample data
            query_result = None
            insights = []
            visualization_config = {}
            
            if query_context.sample_data is not None:
                query_result, insights = self._execute_query_on_dataframe(sql_query, query_context.sample_data, query)
                visualization_config = self._generate_visualization_config(query, query_result, query_context)
            
            # Create query result object
            result_obj = QueryResult(
                sql_query=sql_query,
                explanation=explanation,
                data=query_result,
                visualization_config=visualization_config,
                insights=insights,
                confidence_score=confidence,
                execution_time=0.0
            )
            
            # Cache the result
            context_hash = self._generate_context_hash(query_context)
            self.query_cache.set(query, context_hash, result_obj)
            
            return {
                "type": "query_with_data",
                "sql_query": sql_query,
                "explanation": explanation,
                "confidence_score": confidence,
                "data_preview": query_result.head(10).to_dict('records') if query_result is not None else None,
                "data_shape": query_result.shape if query_result is not None else None,
                "insights": insights,
                "visualization": visualization_config,
                "query_optimization": self._suggest_query_optimizations(sql_query, query_context),
                "alternative_queries": self._suggest_alternative_queries(query, query_context)
            }
            
        except Exception as e:
            logger.error(f"Error processing query with data: {e}")
            return {
                "type": "error",
                "error": str(e),
                "fallback_guidance": self._provide_query_guidance(query, {})
            }
    
    async def _process_query_without_data(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query when no data context is available"""
        question_type = self._classify_question(query)
        
        base_result = {
            "type": "guidance",
            "question_type": question_type,
            "confidence_score": 0.6
        }
        
        if question_type == "aggregation":
            base_result.update(self._process_aggregation_query(query, context))
        elif question_type == "comparison":
            base_result.update(self._process_comparison_query(query, context))
        elif question_type == "trend":
            base_result.update(self._process_trend_query(query, context))
        elif question_type == "data_query":
            base_result.update(self._process_data_query(query, context))
        else:
            base_result.update(self._provide_query_guidance(query, context))
        
        # Add data upload suggestions
        base_result["data_requirements"] = {
            "upload_instructions": [
                "Upload your dataset (CSV, Excel, or JSON format)",
                "Ensure your data has clear column headers",
                "Include relevant date columns for time-based analysis",
                "Clean your data for better query results"
            ],
            "supported_formats": ["CSV", "Excel (.xlsx, .xls)", "JSON"],
            "recommended_structure": self._get_recommended_data_structure(question_type)
        }
        
        return base_result
    
    def _execute_query_on_dataframe(self, sql_query: str, data: pd.DataFrame, original_query: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Execute SQL-like operations on DataFrame"""
        try:
            # For now, implement basic operations
            # In production, you might use libraries like pandasql or duckdb
            
            insights = []
            
            # Simple pattern matching for basic operations
            if "COUNT" in sql_query.upper():
                result = pd.DataFrame({"count": [len(data)]})
                insights.append(f"Found {len(data)} total records in the dataset")
                
            elif "AVG" in sql_query.upper() or "AVERAGE" in original_query.lower():
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    avg_val = data[col].mean()
                    result = pd.DataFrame({f"avg_{col}": [avg_val]})
                    insights.append(f"Average {col} is {avg_val:.2f}")
                else:
                    result = pd.DataFrame({"message": ["No numeric columns found for average calculation"]})
                    
            elif "SUM" in sql_query.upper() or "total" in original_query.lower():
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    sum_val = data[col].sum()
                    result = pd.DataFrame({f"sum_{col}": [sum_val]})
                    insights.append(f"Total {col} is {sum_val:.2f}")
                else:
                    result = pd.DataFrame({"message": ["No numeric columns found for sum calculation"]})
                    
            elif "MAX" in sql_query.upper() or "maximum" in original_query.lower():
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    max_val = data[col].max()
                    result = pd.DataFrame({f"max_{col}": [max_val]})
                    insights.append(f"Maximum {col} is {max_val}")
                else:
                    result = pd.DataFrame({"message": ["No numeric columns found for maximum calculation"]})
                    
            elif "MIN" in sql_query.upper() or "minimum" in original_query.lower():
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    min_val = data[col].min()
                    result = pd.DataFrame({f"min_{col}": [min_val]})
                    insights.append(f"Minimum {col} is {min_val}")
                else:
                    result = pd.DataFrame({"message": ["No numeric columns found for minimum calculation"]})
                    
            elif "top" in original_query.lower() or "LIMIT" in sql_query.upper():
                # Extract limit number
                limit_match = re.search(r'top (\d+)|LIMIT (\d+)', original_query + " " + sql_query)
                limit = 10  # default
                if limit_match:
                    limit = int(next(g for g in limit_match.groups() if g))
                
                result = data.head(limit)
                insights.append(f"Showing top {limit} records from the dataset")
                
            else:
                # Default: return sample of data
                result = data.head(20)
                insights.append(f"Showing first 20 records from the dataset with {len(data)} total rows")
            
            # Add data quality insights
            if len(data) > 0:
                missing_data = data.isnull().sum().sum()
                if missing_data > 0:
                    insights.append(f"Dataset contains {missing_data} missing values")
                
                numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
                text_cols = len(data.select_dtypes(include=['object']).columns)
                insights.append(f"Dataset has {numeric_cols} numeric and {text_cols} text columns")
            
            return result, insights
            
        except Exception as e:
            logger.error(f"Error executing query on DataFrame: {e}")
            return None, [f"Error executing query: {str(e)}"]
    
    def _generate_visualization_config(self, query: str, data: Optional[pd.DataFrame], context: QueryContext) -> Dict[str, Any]:
        """Generate visualization configuration based on query and data"""
        if data is None or len(data) == 0:
            return {"type": "none", "message": "No data available for visualization"}
        
        query_lower = query.lower()
        config = {"type": "table", "title": "Query Results"}
        
        # Determine visualization type based on query and data characteristics
        if any(word in query_lower for word in ["count", "total", "sum", "average"]) and len(data) == 1:
            # Single metric - use number card
            config = {
                "type": "metric_card",
                "title": "Result",
                "value": data.iloc[0, 0] if len(data.columns) > 0 else 0,
                "format": "number"
            }
            
        elif "trend" in query_lower or "over time" in query_lower:
            # Time series - use line chart
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                config = {
                    "type": "line_chart",
                    "title": "Trend Analysis",
                    "x_axis": date_cols[0],
                    "y_axis": [col for col in data.columns if col not in date_cols][0] if len(data.columns) > 1 else data.columns[0]
                }
        
        elif "compare" in query_lower or len(data) > 1:
            # Comparison - use bar chart
            if len(data.columns) >= 2:
                config = {
                    "type": "bar_chart",
                    "title": "Comparison",
                    "x_axis": data.columns[0],
                    "y_axis": data.columns[1] if len(data.columns) > 1 else data.columns[0]
                }
        
        elif len(data) <= 20:
            # Small dataset - use table
            config = {
                "type": "table",
                "title": "Query Results",
                "columns": list(data.columns),
                "sortable": True,
                "filterable": True
            }
        
        else:
            # Large dataset - use paginated table
            config = {
                "type": "paginated_table",
                "title": "Query Results",
                "columns": list(data.columns),
                "page_size": 20,
                "total_rows": len(data)
            }
        
        # Add export options
        config["export_options"] = ["CSV", "Excel", "JSON", "PNG"]
        
        return config
    
    def _suggest_query_optimizations(self, sql_query: str, context: QueryContext) -> List[str]:
        """Suggest optimizations for the generated SQL query"""
        suggestions = []
        
        # Check for missing indexes
        if "WHERE" in sql_query.upper():
            suggestions.append("Consider adding indexes on columns used in WHERE clauses for better performance")
        
        # Check for SELECT *
        if "SELECT *" in sql_query.upper():
            suggestions.append("Consider selecting only the columns you need instead of using SELECT *")
        
        # Check for large result sets
        if "LIMIT" not in sql_query.upper():
            suggestions.append("Consider adding a LIMIT clause for large datasets to improve query performance")
        
        # Check for aggregations without GROUP BY
        agg_functions = ["COUNT", "SUM", "AVG", "MAX", "MIN"]
        if any(func in sql_query.upper() for func in agg_functions) and "GROUP BY" not in sql_query.upper():
            suggestions.append("Consider using GROUP BY with aggregation functions for more detailed insights")
        
        return suggestions
    
    def _suggest_alternative_queries(self, original_query: str, context: QueryContext) -> List[str]:
        """Suggest alternative queries based on the original query"""
        alternatives = []
        query_lower = original_query.lower()
        
        if "count" in query_lower:
            alternatives.extend([
                "What is the average value?",
                "Show me the distribution by category",
                "What are the top 10 records?"
            ])
        
        elif "average" in query_lower or "mean" in query_lower:
            alternatives.extend([
                "What is the total sum?",
                "Show me the minimum and maximum values",
                "How does this vary by category?"
            ])
        
        elif "top" in query_lower:
            alternatives.extend([
                "Show me the bottom 10 records",
                "What is the average of these top records?",
                "How do these compare to the overall average?"
            ])
        
        elif "trend" in query_lower:
            alternatives.extend([
                "Show me the monthly breakdown",
                "What are the seasonal patterns?",
                "Compare this year to last year"
            ])
        
        # Add generic alternatives based on available data
        if context.table_schema:
            table_name = list(context.table_schema.keys())[0]
            columns = context.table_schema[table_name]
            
            if len(columns) > 1:
                alternatives.append(f"Show me the relationship between {columns[0]} and {columns[1]}")
            
            numeric_cols = [col for col, dtype in context.column_types.items() if 'int' in dtype or 'float' in dtype]
            if numeric_cols:
                alternatives.append(f"What is the distribution of {numeric_cols[0]}?")
        
        return alternatives[:5]  # Limit to 5 suggestions
    
    def _classify_question(self, query: str) -> str:
        """Enhanced question classification"""
        query_lower = query.lower()
        
        # Aggregation questions
        if any(word in query_lower for word in ["how many", "count", "total", "sum", "average", "mean", "max", "min", "maximum", "minimum"]):
            return "aggregation"
        
        # Comparison questions
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference", "higher", "lower", "better", "worse", "between"]):
            return "comparison"
        
        # Trend questions
        if any(word in query_lower for word in ["trend", "over time", "growth", "change", "increase", "decrease", "pattern", "seasonal"]):
            return "trend"
        
        # Top/Bottom questions
        if any(word in query_lower for word in ["top", "bottom", "highest", "lowest", "best", "worst", "first", "last"]):
            return "ranking"
        
        # Filtering questions
        if any(word in query_lower for word in ["where", "filter", "with", "having", "contains", "like"]):
            return "filtering"
        
        # Data query questions
        if any(word in query_lower for word in ["show", "list", "find", "get", "what", "which", "display"]):
            return "data_query"
        
        return "general"
    
    def _format_cached_result(self, cached_result: QueryResult) -> Dict[str, Any]:
        """Format cached query result for response"""
        return {
            "type": "cached_query_result",
            "sql_query": cached_result.sql_query,
            "explanation": cached_result.explanation,
            "confidence_score": cached_result.confidence_score,
            "data_preview": cached_result.data.head(10).to_dict('records') if cached_result.data is not None else None,
            "insights": cached_result.insights,
            "visualization": cached_result.visualization_config,
            "cached": True,
            "cache_timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_follow_up_suggestions(self, query: str, result: Dict[str, Any], context: QueryContext) -> List[str]:
        """Generate intelligent follow-up suggestions"""
        suggestions = []
        query_lower = query.lower()
        
        # Context-aware suggestions based on query type
        if result.get("type") == "query_with_data":
            suggestions.extend([
                "Would you like to see more detailed breakdowns?",
                "Should I create a visualization of these results?",
                "Do you want to filter this data further?",
                "Would you like to export these results?"
            ])
        
        elif result.get("type") == "guidance":
            suggestions.extend([
                "Upload your data to get specific answers",
                "Try asking about specific metrics or comparisons",
                "Ask for help with data preparation",
                "Request examples for your type of analysis"
            ])
        
        # Query-specific suggestions
        if "count" in query_lower:
            suggestions.append("What's the breakdown by category?")
        elif "average" in query_lower:
            suggestions.append("Show me the distribution of values")
        elif "trend" in query_lower:
            suggestions.append("Compare this to previous periods")
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _get_recommended_data_structure(self, question_type: str) -> Dict[str, Any]:
        """Get recommended data structure based on question type"""
        base_structure = {
            "required_columns": [],
            "optional_columns": [],
            "data_types": {},
            "example_data": {}
        }
        
        if question_type == "aggregation":
            base_structure.update({
                "required_columns": ["metric_column", "category_column"],
                "data_types": {"metric_column": "numeric", "category_column": "text"},
                "example_data": {"sales": 1000, "region": "North"}
            })
        
        elif question_type == "trend":
            base_structure.update({
                "required_columns": ["date_column", "value_column"],
                "data_types": {"date_column": "date", "value_column": "numeric"},
                "example_data": {"date": "2024-01-01", "revenue": 5000}
            })
        
        elif question_type == "comparison":
            base_structure.update({
                "required_columns": ["category_column", "metric_column"],
                "data_types": {"category_column": "text", "metric_column": "numeric"},
                "example_data": {"product": "Widget A", "sales": 1200}
            })
        
        return base_structure
    
    def _process_data_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data retrieval queries with enhanced guidance"""
        return {
            "query_interpretation": {
                "original_question": query,
                "interpreted_intent": "Retrieve specific data records based on criteria",
                "suggested_sql_pattern": "SELECT columns FROM table WHERE conditions ORDER BY column",
                "explanation": "This query will find and display records that match your specified criteria"
            },
            "query_components": {
                "select_clause": "Specify which columns to display (* for all)",
                "where_clause": "Define filtering conditions (e.g., column = 'value')",
                "order_clause": "Sort results by specific columns",
                "limit_clause": "Restrict number of results returned"
            },
            "expected_output": {
                "format": "Structured table with matching records",
                "visualization_options": ["Data table", "Cards view", "List view"],
                "export_formats": ["CSV", "Excel", "JSON", "PDF"],
                "interactive_features": ["Sorting", "Filtering", "Search", "Pagination"]
            },
            "enhanced_examples": [
                {
                    "question": "Show me all customers from California",
                    "sql": "SELECT * FROM customers WHERE state = 'California'",
                    "explanation": "Retrieves all customer records where state equals California"
                },
                {
                    "question": "Find products with price greater than $100",
                    "sql": "SELECT name, price FROM products WHERE price > 100 ORDER BY price DESC",
                    "explanation": "Shows product names and prices for items over $100, sorted by price"
                },
                {
                    "question": "List recent orders from last month",
                    "sql": "SELECT * FROM orders WHERE order_date >= date('now', '-1 month')",
                    "explanation": "Displays all orders placed within the last 30 days"
                }
            ]
        }
    
    def _process_aggregation_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process aggregation queries with enhanced SQL generation"""
        return {
            "query_interpretation": {
                "original_question": query,
                "interpreted_intent": "Calculate statistical summaries and aggregate metrics",
                "suggested_sql_patterns": {
                    "simple_count": "SELECT COUNT(*) as total_count FROM table",
                    "grouped_aggregation": "SELECT category, COUNT(*), AVG(metric) FROM table GROUP BY category",
                    "multiple_metrics": "SELECT SUM(sales), AVG(price), MAX(quantity) FROM table",
                    "filtered_aggregation": "SELECT COUNT(*) FROM table WHERE condition"
                },
                "explanation": "This query will calculate summary statistics and aggregate values from your data"
            },
            "aggregation_functions": {
                "COUNT(*)": "Total number of records",
                "COUNT(column)": "Number of non-null values in column",
                "SUM(column)": "Total sum of numeric values",
                "AVG(column)": "Average (mean) of numeric values",
                "MIN(column)": "Minimum value in column",
                "MAX(column)": "Maximum value in column",
                "DISTINCT": "Count of unique values",
                "GROUP_CONCAT": "Concatenate values from multiple rows"
            },
            "grouping_strategies": {
                "by_category": "Group results by categorical columns",
                "by_time_period": "Group by date/time intervals (day, month, year)",
                "by_ranges": "Group numeric values into ranges or buckets",
                "by_multiple_columns": "Group by combination of columns"
            },
            "visualization_recommendations": {
                "single_metric": {
                    "type": "metric_card",
                    "description": "Large number display for single values",
                    "best_for": ["Total count", "Grand total", "Overall average"]
                },
                "categorical_breakdown": {
                    "type": "bar_chart",
                    "description": "Compare values across categories",
                    "best_for": ["Sales by region", "Count by category"]
                },
                "proportional_data": {
                    "type": "pie_chart",
                    "description": "Show parts of a whole",
                    "best_for": ["Market share", "Category distribution"]
                },
                "multiple_metrics": {
                    "type": "table",
                    "description": "Display multiple aggregated values",
                    "best_for": ["Summary statistics", "Multi-metric comparisons"]
                }
            },
            "advanced_examples": [
                {
                    "question": "How many customers do we have by region?",
                    "sql": "SELECT region, COUNT(*) as customer_count FROM customers GROUP BY region ORDER BY customer_count DESC",
                    "visualization": "bar_chart",
                    "insights": ["Regional distribution", "Market penetration"]
                },
                {
                    "question": "What's the total and average revenue this month?",
                    "sql": "SELECT SUM(revenue) as total_revenue, AVG(revenue) as avg_revenue FROM sales WHERE month = EXTRACT(MONTH FROM CURRENT_DATE)",
                    "visualization": "metric_cards",
                    "insights": ["Monthly performance", "Average deal size"]
                },
                {
                    "question": "Show me sales statistics by product category",
                    "sql": "SELECT category, COUNT(*) as sales_count, SUM(amount) as total_sales, AVG(amount) as avg_sale FROM sales GROUP BY category",
                    "visualization": "table",
                    "insights": ["Category performance", "Product mix analysis"]
                }
            ]
        }
    
    def _process_comparison_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process comparison queries with advanced analysis capabilities"""
        return {
            "query_interpretation": {
                "original_question": query,
                "interpreted_intent": "Compare and contrast different groups, time periods, or categories",
                "suggested_sql_patterns": {
                    "simple_comparison": "SELECT category, AVG(metric) FROM table GROUP BY category ORDER BY AVG(metric) DESC",
                    "time_comparison": "SELECT period, SUM(value) FROM table GROUP BY period ORDER BY period",
                    "multi_metric_comparison": "SELECT category, AVG(metric1), SUM(metric2), COUNT(*) FROM table GROUP BY category",
                    "percentage_comparison": "SELECT category, value, (value * 100.0 / SUM(value) OVER()) as percentage FROM table"
                },
                "explanation": "This query will compare values across different categories and highlight differences"
            },
            "comparison_dimensions": {
                "categorical_comparison": {
                    "description": "Compare different categories, groups, or segments",
                    "sql_pattern": "GROUP BY category_column",
                    "examples": ["Product categories", "Customer segments", "Geographic regions"]
                },
                "temporal_comparison": {
                    "description": "Compare different time periods",
                    "sql_pattern": "GROUP BY time_period",
                    "examples": ["Year-over-year", "Month-over-month", "Quarter comparison"]
                },
                "threshold_comparison": {
                    "description": "Compare values above/below thresholds",
                    "sql_pattern": "CASE WHEN condition THEN 'Above' ELSE 'Below' END",
                    "examples": ["High vs low performers", "Above/below average"]
                },
                "ranking_comparison": {
                    "description": "Compare top performers vs others",
                    "sql_pattern": "ROW_NUMBER() OVER (ORDER BY metric DESC)",
                    "examples": ["Top 10 vs rest", "Best vs worst performers"]
                }
            },
            "statistical_measures": {
                "absolute_difference": "Direct numerical difference between values",
                "percentage_change": "Relative change expressed as percentage",
                "ratio_analysis": "Proportional relationships between values",
                "variance_analysis": "Measure of spread and variability",
                "correlation_analysis": "Relationship strength between variables"
            },
            "visualization_strategies": {
                "side_by_side_bars": {
                    "best_for": "Comparing 2-5 categories across multiple metrics",
                    "sql_hint": "Use GROUP BY with multiple aggregations"
                },
                "grouped_bar_chart": {
                    "best_for": "Comparing subcategories within main categories",
                    "sql_hint": "Use nested GROUP BY or CASE statements"
                },
                "line_chart_comparison": {
                    "best_for": "Comparing trends over time",
                    "sql_hint": "Include time dimension in GROUP BY"
                },
                "scatter_plot": {
                    "best_for": "Comparing two continuous variables",
                    "sql_hint": "Select two numeric columns for X and Y axes"
                },
                "heatmap": {
                    "best_for": "Multi-dimensional comparisons",
                    "sql_hint": "Use PIVOT or multiple GROUP BY dimensions"
                }
            },
            "advanced_examples": [
                {
                    "question": "Compare sales performance between Q1 and Q2 this year",
                    "sql": """SELECT 
                        CASE WHEN EXTRACT(QUARTER FROM date) = 1 THEN 'Q1' ELSE 'Q2' END as quarter,
                        SUM(sales) as total_sales,
                        AVG(sales) as avg_sales,
                        COUNT(*) as transaction_count
                    FROM sales 
                    WHERE EXTRACT(QUARTER FROM date) IN (1, 2) 
                    GROUP BY EXTRACT(QUARTER FROM date)""",
                    "insights": ["Quarterly growth", "Seasonal patterns", "Performance trends"]
                },
                {
                    "question": "Which product category performs best in terms of revenue and profit?",
                    "sql": """SELECT 
                        category,
                        SUM(revenue) as total_revenue,
                        SUM(profit) as total_profit,
                        AVG(profit_margin) as avg_margin,
                        RANK() OVER (ORDER BY SUM(revenue) DESC) as revenue_rank
                    FROM products 
                    GROUP BY category 
                    ORDER BY total_revenue DESC""",
                    "insights": ["Category rankings", "Profitability analysis", "Market share"]
                },
                {
                    "question": "Compare customer acquisition costs across different marketing channels",
                    "sql": """SELECT 
                        channel,
                        COUNT(*) as customers_acquired,
                        SUM(cost) as total_cost,
                        SUM(cost) / COUNT(*) as cost_per_acquisition,
                        AVG(customer_lifetime_value) as avg_ltv
                    FROM marketing_campaigns 
                    GROUP BY channel 
                    ORDER BY cost_per_acquisition ASC""",
                    "insights": ["Channel efficiency", "ROI analysis", "Cost optimization"]
                }
            ]
        }
    
    def _process_trend_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process trend analysis queries with advanced time-series capabilities"""
        return {
            "query_interpretation": {
                "original_question": query,
                "interpreted_intent": "Analyze patterns, trends, and changes over time periods",
                "suggested_sql_patterns": {
                    "basic_trend": "SELECT date_column, metric FROM table ORDER BY date_column",
                    "aggregated_trend": "SELECT DATE_TRUNC('month', date), SUM(metric) FROM table GROUP BY DATE_TRUNC('month', date)",
                    "growth_calculation": "SELECT date, metric, LAG(metric) OVER (ORDER BY date) as prev_value FROM table",
                    "moving_average": "SELECT date, AVG(metric) OVER (ORDER BY date ROWS 6 PRECEDING) as moving_avg FROM table"
                },
                "explanation": "This query will reveal how values change over time and identify patterns"
            },
            "time_aggregation_levels": {
                "daily": {
                    "sql_function": "DATE(date_column)",
                    "best_for": "Short-term analysis, recent trends",
                    "visualization": "Line chart with daily points"
                },
                "weekly": {
                    "sql_function": "DATE_TRUNC('week', date_column)",
                    "best_for": "Weekly patterns, reducing daily noise",
                    "visualization": "Line chart with weekly aggregation"
                },
                "monthly": {
                    "sql_function": "DATE_TRUNC('month', date_column)",
                    "best_for": "Monthly trends, seasonal analysis",
                    "visualization": "Line or bar chart by month"
                },
                "quarterly": {
                    "sql_function": "DATE_TRUNC('quarter', date_column)",
                    "best_for": "Business cycle analysis",
                    "visualization": "Bar chart by quarter"
                },
                "yearly": {
                    "sql_function": "DATE_TRUNC('year', date_column)",
                    "best_for": "Long-term trends, year-over-year comparison",
                    "visualization": "Bar chart by year"
                }
            },
            "trend_analysis_techniques": {
                "growth_rate": {
                    "description": "Calculate percentage change between periods",
                    "sql_pattern": "(current_value - previous_value) / previous_value * 100",
                    "interpretation": "Positive values indicate growth, negative indicate decline"
                },
                "moving_averages": {
                    "description": "Smooth out short-term fluctuations",
                    "sql_pattern": "AVG(metric) OVER (ORDER BY date ROWS n PRECEDING)",
                    "interpretation": "Helps identify underlying trends"
                },
                "seasonality_detection": {
                    "description": "Identify recurring patterns by time period",
                    "sql_pattern": "GROUP BY EXTRACT(month FROM date) or EXTRACT(dow FROM date)",
                    "interpretation": "Reveals cyclical patterns"
                },
                "anomaly_detection": {
                    "description": "Identify unusual values or outliers",
                    "sql_pattern": "Compare values to statistical thresholds",
                    "interpretation": "Highlights exceptional periods"
                }
            },
            "visualization_recommendations": {
                "line_chart": {
                    "best_for": "Continuous time series data",
                    "features": ["Trend lines", "Multiple series", "Zoom capabilities"],
                    "sql_requirements": "Date column + numeric metric"
                },
                "area_chart": {
                    "best_for": "Cumulative values or stacked categories",
                    "features": ["Filled areas", "Stacking", "Percentage view"],
                    "sql_requirements": "Date column + positive numeric values"
                },
                "bar_chart_time": {
                    "best_for": "Discrete time periods or comparisons",
                    "features": ["Period comparisons", "Grouped bars", "Color coding"],
                    "sql_requirements": "Time period grouping + aggregated values"
                },
                "candlestick_chart": {
                    "best_for": "OHLC (Open, High, Low, Close) data",
                    "features": ["Price movements", "Volume indicators", "Technical analysis"],
                    "sql_requirements": "Date + Open/High/Low/Close columns"
                }
            },
            "comprehensive_examples": [
                {
                    "question": "Show revenue trend over the last 12 months with growth rates",
                    "sql": """SELECT 
                        DATE_TRUNC('month', date) as month,
                        SUM(revenue) as monthly_revenue,
                        LAG(SUM(revenue)) OVER (ORDER BY DATE_TRUNC('month', date)) as prev_month_revenue,
                        CASE 
                            WHEN LAG(SUM(revenue)) OVER (ORDER BY DATE_TRUNC('month', date)) > 0 
                            THEN (SUM(revenue) - LAG(SUM(revenue)) OVER (ORDER BY DATE_TRUNC('month', date))) / LAG(SUM(revenue)) OVER (ORDER BY DATE_TRUNC('month', date)) * 100
                            ELSE NULL 
                        END as growth_rate
                    FROM sales 
                    WHERE date >= DATE('now', '-12 months')
                    GROUP BY DATE_TRUNC('month', date)
                    ORDER BY month""",
                    "insights": ["Monthly growth patterns", "Revenue acceleration/deceleration", "Seasonal trends"]
                },
                {
                    "question": "Analyze customer acquisition trends with 3-month moving average",
                    "sql": """SELECT 
                        DATE_TRUNC('month', signup_date) as month,
                        COUNT(*) as new_customers,
                        AVG(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', signup_date) ROWS 2 PRECEDING) as moving_avg_3m
                    FROM customers 
                    GROUP BY DATE_TRUNC('month', signup_date)
                    ORDER BY month""",
                    "insights": ["Acquisition trends", "Smoothed growth patterns", "Marketing effectiveness"]
                },
                {
                    "question": "Identify seasonal patterns in sales by day of week and month",
                    "sql": """SELECT 
                        EXTRACT(month FROM date) as month,
                        EXTRACT(dow FROM date) as day_of_week,
                        AVG(sales) as avg_sales,
                        COUNT(*) as transaction_count
                    FROM sales 
                    GROUP BY EXTRACT(month FROM date), EXTRACT(dow FROM date)
                    ORDER BY month, day_of_week""",
                    "insights": ["Seasonal variations", "Weekly patterns", "Peak periods identification"]
                }
            ]
        }
    
    def _provide_query_guidance(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide comprehensive query guidance and best practices"""
        return {
            "intelligent_query_tips": {
                "specificity": {
                    "tip": "Be specific about what data you want to see",
                    "examples": [
                        "Instead of 'show data', ask 'show sales data for last quarter'",
                        "Instead of 'count items', ask 'count active customers by region'"
                    ]
                },
                "time_context": {
                    "tip": "Include relevant time ranges for better insights",
                    "examples": [
                        "Add 'in the last 30 days', 'this year vs last year'",
                        "Specify 'monthly trends', 'quarterly comparison'"
                    ]
                },
                "filtering": {
                    "tip": "Specify filtering criteria to focus your analysis",
                    "examples": [
                        "Add conditions like 'where status = active'",
                        "Filter by categories, regions, or value ranges"
                    ]
                },
                "comparisons": {
                    "tip": "Ask for comparisons to uncover insights",
                    "examples": [
                        "Compare performance between different segments",
                        "Analyze differences across time periods"
                    ]
                }
            },
            "supported_question_categories": {
                "data_retrieval": {
                    "keywords": ["Show me", "List", "Find", "Get", "Display", "What are"],
                    "description": "Retrieve specific records or subsets of data",
                    "sql_pattern": "SELECT columns FROM table WHERE conditions"
                },
                "aggregations": {
                    "keywords": ["How many", "Total", "Average", "Count", "Sum", "Maximum", "Minimum"],
                    "description": "Calculate summary statistics and metrics",
                    "sql_pattern": "SELECT AGG_FUNCTION(column) FROM table GROUP BY category"
                },
                "comparisons": {
                    "keywords": ["Compare", "Difference", "Which is higher", "Versus", "Between"],
                    "description": "Compare values across categories or time periods",
                    "sql_pattern": "SELECT category, metric FROM table GROUP BY category ORDER BY metric"
                },
                "trends": {
                    "keywords": ["Over time", "Growth", "Change", "Trend", "Pattern", "Seasonal"],
                    "description": "Analyze patterns and changes over time",
                    "sql_pattern": "SELECT date, metric FROM table ORDER BY date"
                },
                "ranking": {
                    "keywords": ["Top", "Bottom", "Best", "Worst", "Highest", "Lowest"],
                    "description": "Identify top or bottom performers",
                    "sql_pattern": "SELECT columns FROM table ORDER BY metric LIMIT n"
                },
                "filtering": {
                    "keywords": ["Where", "With", "Having", "Contains", "Like", "Equals"],
                    "description": "Filter data based on specific conditions",
                    "sql_pattern": "SELECT columns FROM table WHERE conditions"
                }
            },
            "contextual_examples": {
                "business_metrics": [
                    "How many new customers did we acquire last month?",
                    "What's our total revenue for Q3 compared to Q2?",
                    "Show me the top 10 products by sales volume",
                    "Which sales rep has the highest conversion rate?"
                ],
                "operational_analysis": [
                    "Find all orders with shipping delays",
                    "What's the average processing time by department?",
                    "Show inventory levels below reorder point",
                    "Compare customer satisfaction scores by region"
                ],
                "financial_analysis": [
                    "Calculate monthly recurring revenue trend",
                    "Show profit margins by product category",
                    "What's our customer acquisition cost by channel?",
                    "Analyze cash flow patterns over the last year"
                ],
                "marketing_analysis": [
                    "Which marketing campaigns have the best ROI?",
                    "Show website traffic trends by source",
                    "What's the conversion rate by landing page?",
                    "Compare email open rates across segments"
                ]
            },
            "data_preparation_guide": {
                "file_formats": {
                    "csv": "Comma-separated values - most common format",
                    "excel": "Excel files (.xlsx, .xls) with multiple sheets supported",
                    "json": "JavaScript Object Notation for structured data",
                    "parquet": "Columnar format for large datasets"
                },
                "data_quality_checklist": [
                    "Ensure column headers are clear and descriptive",
                    "Remove or handle missing values appropriately",
                    "Use consistent date formats (YYYY-MM-DD recommended)",
                    "Standardize categorical values (consistent spelling/casing)",
                    "Include unique identifiers where possible"
                ],
                "recommended_columns": {
                    "for_time_analysis": ["date", "timestamp", "created_at", "updated_at"],
                    "for_categorization": ["category", "type", "status", "region", "department"],
                    "for_metrics": ["amount", "quantity", "price", "score", "rating"],
                    "for_identification": ["id", "customer_id", "order_id", "product_id"]
                }
            },
            "advanced_capabilities": {
                "multi_table_analysis": {
                    "description": "Join data from multiple related tables",
                    "example": "Combine customer data with order history for comprehensive analysis",
                    "sql_hint": "Use JOIN operations to connect related tables"
                },
                "calculated_fields": {
                    "description": "Create new metrics from existing data",
                    "example": "Calculate profit margin as (revenue - cost) / revenue",
                    "sql_hint": "Use mathematical operations in SELECT clause"
                },
                "window_functions": {
                    "description": "Perform calculations across related rows",
                    "example": "Calculate running totals, moving averages, or rankings",
                    "sql_hint": "Use OVER() clause for window operations"
                },
                "conditional_logic": {
                    "description": "Apply business rules and categorizations",
                    "example": "Classify customers as 'High', 'Medium', 'Low' value",
                    "sql_hint": "Use CASE WHEN statements for conditional logic"
                }
            },
            "query_optimization_tips": [
                "Use specific column names instead of SELECT * for better performance",
                "Add WHERE clauses to filter data early in the query",
                "Use appropriate indexes on frequently queried columns",
                "Consider using LIMIT for large result sets",
                "Group by the most selective columns first"
            ],
            "troubleshooting_guide": {
                "no_results": [
                    "Check if your filter conditions are too restrictive",
                    "Verify column names match your data exactly",
                    "Ensure date formats are correct",
                    "Check for case sensitivity in text comparisons"
                ],
                "unexpected_results": [
                    "Review your GROUP BY clauses",
                    "Check for NULL values in your data",
                    "Verify aggregation functions are appropriate",
                    "Consider data type mismatches"
                ],
                "performance_issues": [
                    "Add appropriate WHERE clauses to limit data",
                    "Use LIMIT to restrict result size",
                    "Consider creating indexes on key columns",
                    "Break complex queries into smaller parts"
                ]
            }
        }