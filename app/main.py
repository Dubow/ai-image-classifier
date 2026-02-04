from fastapi import FastAPI

app = FastAPI(
    title="Image Analyzer API",
    description="an API for analyzing images and extracting information using machine learning models.",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "Backend is running"}
