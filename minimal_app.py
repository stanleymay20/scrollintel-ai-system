from fastapi import FastAPI

app = FastAPI(title="ScrollIntel Minimal API")

@app.get("/")
def root():
    return {"message": "ScrollIntel is working!", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "healthy"}
