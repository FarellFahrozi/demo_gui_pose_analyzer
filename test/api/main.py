from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import os
from dotenv import load_dotenv

from api.routes import analysis, patients
from api.models.schemas import HealthCheckResponse
from api.services.analyzer import PostureAnalyzerService
from api.services.database import DatabaseService

load_dotenv()

app = FastAPI(
    title="Posture Analysis API",
    description="REST API for KURO Performance Postural Assessment System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router)
app.include_router(patients.router)


@app.get("/", response_model=dict)
async def root():
    return {
        "message": "KURO Performance Postural Assessment API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "operational"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    analyzer = PostureAnalyzerService()
    db = DatabaseService()

    model_loaded = analyzer.is_model_loaded()
    db_connected = db.health_check()

    return HealthCheckResponse(
        status="healthy" if (model_loaded and db_connected) else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        database_connected=db_connected
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))

    print(f"""
    ==============================================================
                                                              
         KURO PERFORMANCE POSTURAL ASSESSMENT API                
                                                              
    ==============================================================

    Server starting on: http://{host}:{port}
    API Documentation: http://{host}:{port}/docs
    Health Check: http://{host}:{port}/health

    Press CTRL+C to stop
    """)

    uvicorn.run(app, host=host, port=port, reload=True)
