from fastapi import FastAPI
import strawberry
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class Query:
    @strawberry.field
    def hello(self, name: str = "World") -> str:
        return f"Hello {name}!"

schema = strawberry.Schema(query=Query)
app = FastAPI(title="ScrollIntel GraphQL API")
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
def root():
    return {"message": "ScrollIntel GraphQL API", "graphql": "/graphql"}

@app.get("/health")
def health():
    return {"status": "healthy", "graphql": "available"}
