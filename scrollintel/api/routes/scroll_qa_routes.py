"""
API routes for ScrollQA engine - natural language data querying.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

from ...engines.scroll_qa_engine import ScrollQAEngine, QueryType
from ...security.auth import get_current_user
from ...security.audit import audit_logger
from ...models.database import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scroll-qa", tags=["ScrollQA"])

# Global engine instance
scroll_qa_engine = None


class QueryRequest(BaseModel):
    """Request model for natural language queries."""
    query: str = Field(..., description="Natural language query")
    datasets: List[str] = Field(default=[], description="List of dataset names to query")
    query_type: str = Field(default=QueryType.SQL_GENERATION, description="Type of query processing")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional query context")


class QueryResponse(BaseModel):
    """Response model for query results."""
    query_type: str
    original_query: str
    results: Dict[str, Any]
    execution_time: str
    datasets_used: List[str]


class IndexDatasetRequest(BaseModel):
    """Request model for dataset indexing."""
    dataset_name: str = Field(..., description="Name of the dataset")
    dataset_path: str = Field(..., description="Path to the dataset file")
    description: Optional[str] = Field(default=None, description="Dataset description")


class SchemaRequest(BaseModel):
    """Request model for schema information."""
    dataset_name: str = Field(..., description="Name of the dataset")


async def get_scroll_qa_engine() -> ScrollQAEngine:
    """Get or create ScrollQA engine instance."""
    global scroll_qa_engine
    
    if scroll_qa_engine is None:
        scroll_qa_engine = ScrollQAEngine()
        await scroll_qa_engine.start()
    
    return scroll_qa_engine


@router.post("/query", response_model=QueryResponse)
async def process_natural_language_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """
    Process a natural language query against datasets.
    
    Supports multiple query types:
    - sql_generation: Convert natural language to SQL
    - semantic_search: Vector-based semantic search
    - context_aware: Combined SQL and semantic search
    - multi_source: Query across multiple datasets
    """
    try:
        # Prepare input data
        input_data = {
            "action": "query",
            "query": request.query,
            "datasets": request.datasets,
            "query_type": request.query_type,
            "context": request.context or {}
        }
        
        # Process query
        result = await engine.execute(input_data)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="scroll_qa_query",
            resource_type="query",
            resource_id=request.query[:50],
            details={
                "query_type": request.query_type,
                "datasets": request.datasets,
                "success": True
            }
        )
        
        return QueryResponse(
            query_type=result["query_type"],
            original_query=result["original_query"],
            results=result,
            execution_time=result["execution_time"],
            datasets_used=result.get("datasets_used", request.datasets)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        
        # Audit log error
        await audit_logger.log(
            user_id=current_user.id,
            action="scroll_qa_query",
            resource_type="query",
            resource_id=request.query[:50],
            details={
                "query_type": request.query_type,
                "datasets": request.datasets,
                "success": False,
                "error": str(e)
            },
            success=False,
            error_message=str(e)
        )
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sql-query")
async def generate_sql_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Generate SQL query from natural language."""
    
    # Force SQL generation query type
    request.query_type = QueryType.SQL_GENERATION
    
    return await process_natural_language_query(request, current_user, engine)


@router.post("/semantic-search")
async def semantic_search(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Perform semantic search across indexed datasets."""
    
    # Force semantic search query type
    request.query_type = QueryType.SEMANTIC_SEARCH
    
    return await process_natural_language_query(request, current_user, engine)


@router.post("/context-aware-query")
async def context_aware_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Process context-aware query with data source integration."""
    
    # Force context-aware query type
    request.query_type = QueryType.CONTEXT_AWARE
    
    return await process_natural_language_query(request, current_user, engine)


@router.post("/multi-source-query")
async def multi_source_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Query across multiple data sources."""
    
    if not request.datasets:
        raise HTTPException(status_code=400, detail="At least one dataset must be specified for multi-source query")
    
    # Force multi-source query type
    request.query_type = QueryType.MULTI_SOURCE
    
    return await process_natural_language_query(request, current_user, engine)


@router.post("/index-dataset")
async def index_dataset(
    request: IndexDatasetRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Index a dataset for semantic search."""
    
    try:
        # Prepare input data
        input_data = {
            "action": "index_dataset",
            "dataset_name": request.dataset_name,
            "dataset_path": request.dataset_path,
            "description": request.description
        }
        
        # Process indexing (run in background for large datasets)
        def index_task():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(engine.execute(input_data))
                logger.info(f"Dataset indexing completed: {result}")
            except Exception as e:
                logger.error(f"Dataset indexing failed: {e}")
            finally:
                loop.close()
        
        background_tasks.add_task(index_task)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="index_dataset",
            resource_type="dataset",
            resource_id=request.dataset_name,
            details={
                "dataset_path": request.dataset_path,
                "description": request.description
            }
        )
        
        return {
            "status": "indexing_started",
            "dataset_name": request.dataset_name,
            "message": "Dataset indexing started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting dataset indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema/{dataset_name}")
async def get_dataset_schema(
    dataset_name: str,
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Get schema information for a dataset."""
    
    try:
        input_data = {
            "action": "get_schema",
            "dataset_name": dataset_name
        }
        
        result = await engine.execute(input_data)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_available_datasets(
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """List all available datasets and their schemas."""
    
    try:
        status = engine.get_status()
        
        return {
            "datasets": list(engine.dataset_schemas.keys()) if hasattr(engine, 'dataset_schemas') else [],
            "total_datasets": status.get("datasets_indexed", 0),
            "engine_status": status
        }
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_query_cache(
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Clear the query result cache."""
    
    try:
        input_data = {"action": "clear_cache"}
        result = await engine.execute(input_data)
        
        # Audit log
        await audit_logger.log(
            user_id=current_user.id,
            action="clear_cache",
            resource_type="cache",
            resource_id="scroll_qa_cache",
            details=result
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_engine_status(
    current_user: User = Depends(get_current_user),
    engine: ScrollQAEngine = Depends(get_scroll_qa_engine)
):
    """Get ScrollQA engine status and health information."""
    
    try:
        status = engine.get_status()
        metrics = engine.get_metrics()
        
        return {
            "status": status,
            "metrics": metrics,
            "health_check": await engine.health_check()
        }
        
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-types")
async def get_supported_query_types():
    """Get list of supported query types."""
    
    return {
        "query_types": [
            {
                "type": QueryType.SQL_GENERATION,
                "description": "Convert natural language to SQL queries"
            },
            {
                "type": QueryType.SEMANTIC_SEARCH,
                "description": "Vector-based semantic search across indexed data"
            },
            {
                "type": QueryType.CONTEXT_AWARE,
                "description": "Combined SQL and semantic search with context"
            },
            {
                "type": QueryType.MULTI_SOURCE,
                "description": "Query across multiple datasets simultaneously"
            }
        ]
    }


# Example queries endpoint for documentation
@router.get("/examples")
async def get_example_queries():
    """Get example queries for different query types."""
    
    return {
        "examples": {
            QueryType.SQL_GENERATION: [
                "Show me all customers from New York",
                "What are the top 10 products by sales?",
                "Calculate average order value by month",
                "Find customers who haven't ordered in the last 6 months"
            ],
            QueryType.SEMANTIC_SEARCH: [
                "Find information about customer satisfaction",
                "What do we know about product quality issues?",
                "Show me insights about market trends",
                "Find data related to seasonal patterns"
            ],
            QueryType.CONTEXT_AWARE: [
                "Analyze customer behavior and provide insights",
                "What factors influence our sales performance?",
                "Explain the relationship between marketing spend and revenue",
                "Provide a comprehensive analysis of our top customers"
            ],
            QueryType.MULTI_SOURCE: [
                "Compare sales data across all regions",
                "Analyze customer data from multiple databases",
                "Find correlations between different data sources",
                "Generate a unified view of our business metrics"
            ]
        }
    }