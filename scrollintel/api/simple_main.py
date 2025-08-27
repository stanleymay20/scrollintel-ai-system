from fastapi import FastAPI
import os

# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI(title="ScrollIntel API")

@app.get("/")
def root():
    return {"message": "ScrollIntel is running!", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/test")
def test():
    return {"test": "success", "tensorflow": "loaded"}
