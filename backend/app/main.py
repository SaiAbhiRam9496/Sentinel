from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.health import router as health_router
from backend.app.api.upload import router as upload_router
import logging
from contextlib import asynccontextmanager
from backend.app.db.session import Base, engine, check_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel")

from sqlalchemy import text

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    try:
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS schema_details JSON;"))
            conn.execute(text("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS match_details JSON;"))
            conn.commit()
        logger.info("Database tables initialized and updated successfully.")
    except Exception as e:
        logger.error(f"Failed to create/update database tables: {e}")
    yield


app = FastAPI(
    title="Sentinal",
    description="Enterprise Data Quality, Profiling, Anomaly Detection & RAG Analytics",
    version="1.0.0",
    lifespan=lifespan
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list + ["*"] if settings.ENVIRONMENT == "development" else settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(health_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "Welcome to Sentinel API",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
