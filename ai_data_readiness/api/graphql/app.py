"""GraphQL application setup."""

from fastapi import FastAPI, Depends, Request
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL
import strawberry

from .schema import schema
from ..middleware.auth import get_current_user


class AuthContext:
    """GraphQL context with authentication."""
    
    def __init__(self, request: Request, user: dict = None):
        self.request = request
        self.user = user


async def get_context(
    request: Request,
    user: dict = Depends(get_current_user)
) -> AuthContext:
    """Create GraphQL context with authenticated user."""
    return AuthContext(request=request, user=user)


# Create GraphQL router
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ],
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