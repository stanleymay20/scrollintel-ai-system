"""
GraphQL application for Advanced Analytics Dashboard System.
"""
from fastapi import FastAPI, Depends
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from .analytics_schema import schema
from ...security.auth import get_current_user
from ...core.rate_limiter import rate_limit


# Create GraphQL router with authentication
async def get_context(current_user: dict = Depends(get_current_user)):
    """Get GraphQL context with authenticated user."""
    return {
        "user": current_user,
        "user_id": current_user["id"]
    }


# Create GraphQL router
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ],
)


def create_graphql_app() -> FastAPI:
    """Create FastAPI app with GraphQL endpoint."""
    app = FastAPI(
        title="Analytics Dashboard GraphQL API",
        description="Flexible GraphQL API for Advanced Analytics Dashboard System",
        version="1.0.0"
    )
    
    # Add GraphQL endpoint
    app.include_router(graphql_app, prefix="/graphql")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "graphql-api",
            "version": "1.0.0"
        }
    
    return app


# Export the app
app = create_graphql_app()