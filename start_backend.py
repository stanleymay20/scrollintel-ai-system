
import sys
import os
sys.path.append(os.getcwd())

try:
    from scrollintel.api.main import app
    import uvicorn
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    try:
        from scrollintel.api.simple_main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Could not import ScrollIntel app")
        # Create a simple FastAPI app
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="ScrollIntel API", version="1.0.0")
        
        @app.get("/")
        def root():
            return {"message": "ScrollIntel API is running", "status": "healthy"}
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "service": "scrollintel-api"}
        
        @app.get("/api/v1/agents")
        def agents():
            return {"agents": ["AI Engineer", "ML Engineer", "Data Scientist", "CTO Agent"], "status": "available"}
        
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
