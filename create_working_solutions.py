#!/usr/bin/env python3
"""
Create working Docker solutions.
"""

import os

def create_minimal_solution():
    """Create minimal working solution."""
    
    # Minimal Dockerfile
    dockerfile_content = '''FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install fastapi uvicorn
COPY minimal_app.py .
EXPOSE 8000
CMD ["uvicorn", "minimal_app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open("Dockerfile.minimal", "w") as f:
        f.write(dockerfile_content)
    
    # Minimal app
    app_content = '''from fastapi import FastAPI

app = FastAPI(title="ScrollIntel Minimal API")

@app.get("/")
def root():
    return {"message": "ScrollIntel is working!", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "healthy"}
'''
    
    with open("minimal_app.py", "w") as f:
        f.write(app_content)
    
    print("✅ Created minimal solution")

def create_graphql_solution():
    """Create GraphQL solution."""
    
    # GraphQL Dockerfile
    dockerfile_content = '''FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y curl gcc g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY graphql_requirements.txt .
RUN pip install --upgrade pip && pip install -r graphql_requirements.txt
COPY graphql_app.py .
EXPOSE 8000
CMD ["uvicorn", "graphql_app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open("Dockerfile.graphql", "w") as f:
        f.write(dockerfile_content)
    
    # GraphQL requirements
    requirements_content = '''fastapi>=0.109.0
uvicorn[standard]>=0.27.0
strawberry-graphql[fastapi]>=0.219.0
'''
    
    with open("graphql_requirements.txt", "w") as f:
        f.write(requirements_content)
    
    # GraphQL app
    app_content = '''from fastapi import FastAPI
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
'''
    
    with open("graphql_app.py", "w") as f:
        f.write(app_content)
    
    print("✅ Created GraphQL solution")

def main():
    """Main function."""
    print("🚀 Creating Working Docker Solutions")
    print("=" * 40)
    
    create_minimal_solution()
    create_graphql_solution()
    
    print("\n📋 Solutions Created:")
    print("\n1️⃣ Minimal (No TensorFlow issues):")
    print("   docker build -f Dockerfile.minimal -t scrollintel:minimal .")
    print("   docker run -p 8000:8000 scrollintel:minimal")
    
    print("\n2️⃣ Modern GraphQL:")
    print("   docker build -f Dockerfile.graphql -t scrollintel:graphql .")
    print("   docker run -p 8000:8000 scrollintel:graphql")
    
    print("\n🎯 Both solutions avoid the TensorFlow/Keras compatibility issue!")

if __name__ == "__main__":
    main()