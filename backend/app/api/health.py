from fastapi import APIRouter
from backend.app.db.session import check_db_connection
from backend.app.config import settings
from datetime import datetime, timezone

router = APIRouter(tags=["Health"])

@router.get("/health")
def get_health():
    db_ok = check_db_connection()
    return {
        "status": "healthy" if db_ok else "degraded",
        "app": "Sentinal",
        "version": "1.0.0",
        "database": "connected" if db_ok else "disconnected",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
