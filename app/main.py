from fastapi import FastAPI
from app.api.v1.predict import router as predict_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Image Analyzer API",
    description="an API for analyzing images and extracting information using machine learning models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API is running"}

# Register API v1 routes
app.include_router(predict_router, prefix="/api/v1")

