from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router

app = FastAPI(
    title="BurnoutRadar API",
    description="API for Predicting Employee Burnout Risk",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "message": "Welcome to the BurnoutRadar API. Visit /docs for documentation.",
        "status": "Healthy"
    }
