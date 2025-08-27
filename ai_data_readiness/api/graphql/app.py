"""Modern GraphQL application setup with enhanced features."""

from fastapi import FastAPI, Depends, Request, HTTPException
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL
from strawberry.types import Info
import strawberry
import logging
from typing import Optional

from .schema import schema
from ..middleware.auth import get_current_user

logger = logging.getLogger(__name__)


@strawberry.type
class AuthContext:
    """Modern GraphQL context with enhanced authentication and request info."""
    
    def __init__(self, request: Request, user: Optional[dict] = None):
        self.request = request
        self.user = user
        self.user_id = user.get("id") if user else None
        self.is_authenticated = user is not None
        self.permissions = user.get("permissions", []) if user else []
    
    def require_auth(self) -> dict:
        """Require authentication and return user."""
        if not self.is_authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")
        return self.user
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def require_permission(self, permission: str) -> None:
        """Require specific permission."""
        if not self.has_permission(permission):
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")


async def get_context(
    request: Request,
    user: Optional[dict] = Depends(get_current_user)
) -> AuthContext:
    """Create modern GraphQL context with enhanced features."""
    return AuthContext(request=request, user=user)


# Create modern GraphQL router with enhanced configuration
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ],
    # Enable GraphQL introspection (disable in production)
    introspection=True,
    # Enable GraphQL playground (disable in production)
    graphql_ide="graphiql",
)


def setup_graphql(app: FastAPI):
    """Set up GraphQL endpoint in FastAPI app."""
    # Add GraphQL endpoint
    app.include_router(
        graphql_app,
        prefix="/graphql",
        tags=["graphql"]
    )
    
    # Add GraphQL playground (development only)
    @app.get("/graphql/playground")
    async def graphql_playground():
        """GraphQL playground for development."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GraphQL Playground</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/css/index.css" />
        </head>
        <body>
            <div id="root">
                <style>
                    body { margin: 0; font-family: Open Sans, sans-serif; overflow: hidden; }
                    #root { height: 100vh; }
                </style>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/js/middleware.js"></script>
            <script>
                window.addEventListener('load', function (event) {
                    GraphQLPlayground.init(document.getElementById('root'), {
                        endpoint: '/graphql',
                        subscriptionEndpoint: '/graphql',
                        settings: {
                            'request.credentials': 'include',
                        }
                    })
                })
            </script>
        </body>
        </html>
        """