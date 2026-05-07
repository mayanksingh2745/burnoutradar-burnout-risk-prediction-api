from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="BurnoutRadar API",
    description="API for predicting burnout risk in remote workers using an ML ensemble.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

@app.get("/", tags=["System"])
def root():
    return {"message": "Welcome to BurnoutRadar API. Visit /docs for the API reference."}
