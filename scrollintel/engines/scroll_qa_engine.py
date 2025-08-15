"""
ScrollQA Engine for natural language data querying.
Implements requirements 2.1, 2.2: Natural language to SQL conversion and semantic search.
"""

import os
import json
import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

# AI and NLP dependencies
import openai

# Optional langchain dependencies
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Pinecone as LangchainPinecone
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    OpenAIEmbeddings = None
    LangchainPinecone = None
    RecursiveCharacterTextSplitter = None
    Document = None

# Vector database
try:
    import pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    pinecone = None

# Redis for caching
import redis
import pickle

from .base_engine import BaseEngine, EngineCapability, EngineStatus
from ..core.config import get_config
from ..models.database import Dataset

logger = logging.getLogger(__name__)


class QueryType:
    """Types of queries supported by ScrollQA."""
    SQL_GENERATION = "sql_generation"
    SEMANTIC_SEARCH = "semantic_search"
    CONTEXT_AWARE = "context_aware"
    MULTI_SOURCE = "multi_source"


class ScrollQAEngine(BaseEngine):
    """
    ScrollQA engine for natural language data querying.
    
    Capabilities:
    - Natural language to SQL conversion
    - Vector similarity search using embeddings
    - Context-aware response generation
    - Multi-source data querying
    - Query result caching with Redis
    """
    
    def __init__(self):
        super().__init__(
            engine_id="scroll_qa_engine",
            name="ScrollQA Engine",
            capabilities=[
                EngineCapability.DATA_ANALYSIS,
                EngineCapability.VISUALIZATION
            ]
        )
        
        self.config = get_config()
        self.embeddings = None
        self.vector_store = None
        self.redis_client = None
        self.openai_client = None
        
        # SQL generation templates
        self.sql_templates = {
            "select": "SELECT {columns} FROM {table} WHERE {conditions}",
            "aggregate": "SELECT {group_by}, {aggregation} FROM {table} GROUP BY {group_by}",
            "join": "SELECT {columns} FROM {table1} JOIN {table2} ON {join_condition}",
            "time_series": "SELECT {date_column}, {value_column} FROM {table} WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}' ORDER BY {date_column}"
        }
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 1000
        
        # Query context
        self.query_context = {}
        self.dataset_schemas = {}
        
    async def initialize(self) -> None:
        """Initialize the ScrollQA engine."""
        logger.info("Initializing ScrollQA engine...")
        
        try:
            # Initialize OpenAI client
            if self.config.ai_services.openai_api_key:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.config.ai_services.openai_api_key
                )
                
                # Initialize embeddings if langchain is available
                if HAS_LANGCHAIN and OpenAIEmbeddings:
                    self.embeddings = OpenAIEmbeddings(
                        openai_api_key=self.config.ai_services.openai_api_key,
                        model="text-embedding-ada-002"
                    )
                else:
                    logger.warning("LangChain not available - semantic search will be limited")
            else:
                logger.warning("OpenAI API key not configured - some features will be limited")
            
            # Initialize vector database
            await self._initialize_vector_store()
            
            # Initialize Redis cache
            await self._initialize_redis()
            
            # Load dataset schemas
            await self._load_dataset_schemas()
            
            self.status = EngineStatus.READY
            logger.info("ScrollQA engine initialized successfully")
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Failed to initialize ScrollQA engine: {e}")
            raise
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """
        Process natural language query request.
        
        Args:
            input_data: Dictionary containing query and context
            parameters: Additional processing parameters
            
        Returns:
            Query results and response
        """
        try:
            action = input_data.get("action", "query")
            
            if action == "query":
                return await self._process_query(input_data, parameters)
            elif action == "index_dataset":
                return await self._index_dataset(input_data, parameters)
            elif action == "get_schema":
                return await self._get_dataset_schema(input_data, parameters)
            elif action == "clear_cache":
                return await self._clear_cache(input_data, parameters)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error in ScrollQA processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up ScrollQA engine...")
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.query_context.clear()
        self.dataset_schemas.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "healthy": True,
            "openai_configured": bool(self.openai_client),
            "vector_store_configured": bool(self.vector_store),
            "redis_configured": bool(self.redis_client),
            "datasets_indexed": len(self.dataset_schemas),
            "cache_size": len(self.query_context)
        }
    
    async def _initialize_vector_store(self) -> None:
        """Initialize vector database for semantic search."""
        
        if not HAS_PINECONE or not HAS_LANGCHAIN or not self.config.database.pinecone_api_key:
            logger.warning("Pinecone or LangChain not configured - semantic search will be limited")
            return
        
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=self.config.database.pinecone_api_key,
                environment=self.config.database.pinecone_environment
            )
            
            # Create or connect to index
            index_name = "scrollqa-index"
            
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            # Initialize vector store
            if self.embeddings and LangchainPinecone:
                self.vector_store = LangchainPinecone.from_existing_index(
                    index_name=index_name,
                    embedding=self.embeddings
                )
                logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis client for caching."""
        
        try:
            self.redis_client = redis.asyncio.Redis(
                host=self.config.database.redis_host,
                port=self.config.database.redis_port,
                password=self.config.database.redis_password,
                db=self.config.database.redis_db,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None
    
    async def _load_dataset_schemas(self) -> None:
        """Load schemas for all available datasets."""
        
        try:
            # This would typically load from database
            # For now, we'll implement a basic schema loader
            
            # Load from PostgreSQL if available
            if self.config.database.postgres_url:
                engine = create_engine(self.config.database.postgres_url)
                inspector = inspect(engine)
                
                for table_name in inspector.get_table_names():
                    columns = inspector.get_columns(table_name)
                    self.dataset_schemas[table_name] = {
                        "columns": [
                            {
                                "name": col["name"],
                                "type": str(col["type"]),
                                "nullable": col["nullable"]
                            }
                            for col in columns
                        ],
                        "source_type": "database",
                        "table_name": table_name
                    }
                
                logger.info(f"Loaded schemas for {len(self.dataset_schemas)} datasets")
            
        except Exception as e:
            logger.error(f"Failed to load dataset schemas: {e}")
    
    async def _process_query(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a natural language query."""
        
        query = input_data.get("query", "")
        datasets = input_data.get("datasets", [])
        query_type = input_data.get("query_type", QueryType.SQL_GENERATION)
        
        if not query:
            raise ValueError("Query is required")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, datasets, query_type)
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result:
            logger.info("Returning cached query result")
            return cached_result
        
        # Process query based on type
        if query_type == QueryType.SQL_GENERATION:
            result = await self._generate_sql_query(query, datasets)
        elif query_type == QueryType.SEMANTIC_SEARCH:
            result = await self._semantic_search(query, datasets)
        elif query_type == QueryType.CONTEXT_AWARE:
            result = await self._context_aware_query(query, datasets)
        elif query_type == QueryType.MULTI_SOURCE:
            result = await self._multi_source_query(query, datasets)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        # Cache result
        await self._cache_result(cache_key, result)
        
        return result
    
    async def _generate_sql_query(self, query: str, datasets: List[str]) -> Dict[str, Any]:
        """Generate SQL query from natural language."""
        
        if not self.openai_client:
            raise RuntimeError("OpenAI client not configured")
        
        # Get relevant dataset schemas
        relevant_schemas = {}
        for dataset in datasets:
            if dataset in self.dataset_schemas:
                relevant_schemas[dataset] = self.dataset_schemas[dataset]
        
        if not relevant_schemas:
            raise ValueError("No valid datasets found")
        
        # Create schema context
        schema_context = self._create_schema_context(relevant_schemas)
        
        # Generate SQL using GPT-4
        system_prompt = f"""
You are an expert SQL query generator. Given a natural language question and database schema, generate a precise SQL query.

Database Schema:
{schema_context}

Rules:
1. Generate only valid SQL queries
2. Use proper table and column names from the schema
3. Include appropriate WHERE clauses for filtering
4. Use JOINs when querying multiple tables
5. Add ORDER BY and LIMIT when appropriate
6. Return only the SQL query, no explanations

Example formats:
- Simple query: SELECT column1, column2 FROM table WHERE condition
- Aggregation: SELECT column, COUNT(*) FROM table GROUP BY column
- Join: SELECT t1.col, t2.col FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.ai_services.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate SQL for: {query}"}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            # Execute the query
            query_results = await self._execute_sql_query(sql_query, datasets[0] if datasets else None)
            
            return {
                "query_type": QueryType.SQL_GENERATION,
                "original_query": query,
                "generated_sql": sql_query,
                "results": query_results,
                "datasets_used": datasets,
                "execution_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            raise
    
    async def _semantic_search(self, query: str, datasets: List[str]) -> Dict[str, Any]:
        """Perform semantic search using vector embeddings."""
        
        if not self.vector_store:
            # Fallback to simple text search if vector store not available
            logger.warning("Vector store not configured, using fallback search")
            return await self._fallback_text_search(query, datasets)
        
        try:
            # Perform similarity search
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                query,
                k=10
            )
            
            # Extract relevant information
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'score', 0.0)
                })
            
            # Generate context-aware response
            context = "\n".join([doc.page_content for doc in docs[:5]])
            response = await self._generate_contextual_response(query, context)
            
            return {
                "query_type": QueryType.SEMANTIC_SEARCH,
                "original_query": query,
                "search_results": results,
                "contextual_response": response,
                "datasets_searched": datasets,
                "execution_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise
    
    async def _context_aware_query(self, query: str, datasets: List[str]) -> Dict[str, Any]:
        """Process context-aware query with data source integration."""
        
        # Combine SQL generation with semantic search
        sql_result = await self._generate_sql_query(query, datasets)
        
        if self.vector_store:
            semantic_result = await self._semantic_search(query, datasets)
            
            # Combine results
            combined_context = {
                "sql_data": sql_result.get("results", {}),
                "semantic_context": semantic_result.get("search_results", [])
            }
            
            # Generate comprehensive response
            response = await self._generate_comprehensive_response(query, combined_context)
            
            return {
                "query_type": QueryType.CONTEXT_AWARE,
                "original_query": query,
                "sql_results": sql_result,
                "semantic_results": semantic_result,
                "comprehensive_response": response,
                "datasets_used": datasets,
                "execution_time": datetime.utcnow().isoformat()
            }
        else:
            return sql_result
    
    async def _multi_source_query(self, query: str, datasets: List[str]) -> Dict[str, Any]:
        """Query across multiple data sources."""
        
        results = {}
        
        for dataset in datasets:
            try:
                # Query each dataset individually
                dataset_result = await self._generate_sql_query(query, [dataset])
                results[dataset] = dataset_result
                
            except Exception as e:
                logger.error(f"Error querying dataset {dataset}: {e}")
                results[dataset] = {"error": str(e)}
        
        # Combine and analyze results
        combined_analysis = await self._analyze_multi_source_results(query, results)
        
        return {
            "query_type": QueryType.MULTI_SOURCE,
            "original_query": query,
            "individual_results": results,
            "combined_analysis": combined_analysis,
            "datasets_queried": datasets,
            "execution_time": datetime.utcnow().isoformat()
        }
    
    async def _execute_sql_query(self, sql_query: str, dataset: str = None) -> Dict[str, Any]:
        """Execute SQL query against database."""
        
        try:
            # Use PostgreSQL connection
            engine = create_engine(self.config.database.postgres_url)
            
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                
                # Convert to DataFrame for easier handling
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                return {
                    "data": df.to_dict('records'),
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "query_executed": sql_query
                }
                
        except SQLAlchemyError as e:
            logger.error(f"SQL execution error: {e}")
            return {
                "error": str(e),
                "query_executed": sql_query
            }
    
    async def _generate_contextual_response(self, query: str, context: str) -> str:
        """Generate contextual response using AI."""
        
        if not self.openai_client:
            return "AI response generation not available"
        
        try:
            system_prompt = """
You are a data analyst assistant. Given a user query and relevant context from data sources, 
provide a clear, informative response that answers the user's question based on the available data.

Rules:
1. Be concise and accurate
2. Reference specific data points when available
3. Acknowledge limitations if data is incomplete
4. Provide actionable insights when possible
"""
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.ai_services.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def _generate_comprehensive_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate comprehensive response combining multiple data sources."""
        
        if not self.openai_client:
            return "AI response generation not available"
        
        try:
            # Format context for AI
            formatted_context = json.dumps(context, indent=2, default=str)
            
            system_prompt = """
You are an expert data analyst. Given a user query and comprehensive data context from multiple sources,
provide a detailed analysis that synthesizes information from all available sources.

Rules:
1. Analyze both structured data (SQL results) and unstructured context
2. Identify patterns, trends, and insights
3. Provide specific recommendations when appropriate
4. Acknowledge data limitations and uncertainties
5. Structure your response clearly with key findings
"""
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.ai_services.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nData Context: {formatted_context}"}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating comprehensive response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def _analyze_multi_source_results(self, query: str, results: Dict[str, Any]) -> str:
        """Analyze results from multiple data sources."""
        
        if not self.openai_client:
            return "Multi-source analysis not available"
        
        try:
            # Format results for analysis
            formatted_results = json.dumps(results, indent=2, default=str)
            
            system_prompt = """
You are a data integration specialist. Given query results from multiple data sources,
provide a comprehensive analysis that identifies:

1. Common patterns across sources
2. Discrepancies or conflicts in data
3. Complementary insights from different sources
4. Overall conclusions and recommendations
5. Data quality observations

Be specific about which sources provide which insights.
"""
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.ai_services.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nMulti-source Results: {formatted_results}"}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing multi-source results: {e}")
            return f"Error in analysis: {str(e)}"
    
    def _create_schema_context(self, schemas: Dict[str, Any]) -> str:
        """Create formatted schema context for AI."""
        
        context_parts = []
        
        for dataset_name, schema in schemas.items():
            context_parts.append(f"Table: {dataset_name}")
            context_parts.append("Columns:")
            
            for column in schema.get("columns", []):
                nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
                context_parts.append(f"  - {column['name']} ({column['type']}) {nullable}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and validate SQL query."""
        
        # Remove markdown formatting if present
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Remove comments
        lines = sql_query.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip().startswith('--'):
                cleaned_lines.append(line)
        
        sql_query = '\n'.join(cleaned_lines).strip()
        
        # Ensure query ends with semicolon
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query
    
    def _generate_cache_key(self, query: str, datasets: List[str], query_type: str) -> str:
        """Generate cache key for query."""
        
        key_data = {
            "query": query.lower().strip(),
            "datasets": sorted(datasets),
            "query_type": query_type
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"scrollqa:{cache_key}")
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Error retrieving cached result: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache query result."""
        
        if not self.redis_client:
            return
        
        try:
            cached_data = pickle.dumps(result)
            await self.redis_client.setex(
                f"scrollqa:{cache_key}",
                self.cache_ttl,
                cached_data
            )
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
    
    async def _index_dataset(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Index a dataset for semantic search."""
        
        if not self.vector_store or not HAS_LANGCHAIN:
            raise RuntimeError("Vector store or LangChain not configured")
        
        dataset_name = input_data.get("dataset_name")
        dataset_path = input_data.get("dataset_path")
        
        if not dataset_name or not dataset_path:
            raise ValueError("dataset_name and dataset_path are required")
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)  # Simplified - would handle multiple formats
            
            # Create documents for indexing
            documents = []
            
            # Use text splitter if available
            if RecursiveCharacterTextSplitter:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
            else:
                text_splitter = None
            
            # Convert each row to a document
            for idx, row in df.iterrows():
                # Create text representation of the row
                row_text = f"Dataset: {dataset_name}\n"
                for col, val in row.items():
                    row_text += f"{col}: {val}\n"
                
                # Split if too long and splitter available
                if text_splitter:
                    chunks = text_splitter.split_text(row_text)
                else:
                    chunks = [row_text]
                
                for chunk in chunks:
                    if Document:
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "dataset": dataset_name,
                                "row_index": idx,
                                "source": dataset_path
                            }
                        ))
            
            # Add to vector store
            if documents:
                await asyncio.to_thread(
                    self.vector_store.add_documents,
                    documents
                )
            
            return {
                "dataset_name": dataset_name,
                "documents_indexed": len(documents),
                "rows_processed": len(df),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error indexing dataset: {e}")
            raise
    
    async def _get_dataset_schema(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get schema information for a dataset."""
        
        dataset_name = input_data.get("dataset_name")
        
        if dataset_name in self.dataset_schemas:
            return {
                "dataset_name": dataset_name,
                "schema": self.dataset_schemas[dataset_name]
            }
        else:
            return {
                "dataset_name": dataset_name,
                "error": "Dataset not found"
            }
    
    async def _clear_cache(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clear query cache."""
        
        if not self.redis_client:
            return {"status": "Cache not configured"}
        
        try:
            # Clear all ScrollQA cache keys
            keys = await self.redis_client.keys("scrollqa:*")
            if keys:
                await self.redis_client.delete(*keys)
            
            return {
                "status": "success",
                "keys_cleared": len(keys)
            }
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _fallback_text_search(self, query: str, datasets: List[str]) -> Dict[str, Any]:
        """Fallback text search when vector store is not available."""
        
        # Simple keyword-based search as fallback
        search_terms = query.lower().split()
        
        # Mock search results for demonstration
        results = [
            {
                "content": f"Fallback search result for query: {query}",
                "metadata": {"dataset": dataset, "search_type": "fallback"},
                "relevance_score": 0.5
            }
            for dataset in datasets
        ]
        
        response = f"Fallback search completed for query: {query}. Vector search not available."
        
        return {
            "query_type": QueryType.SEMANTIC_SEARCH,
            "original_query": query,
            "search_results": results,
            "contextual_response": response,
            "datasets_searched": datasets,
            "execution_time": datetime.utcnow().isoformat(),
            "note": "Using fallback search - vector store not configured"
        }