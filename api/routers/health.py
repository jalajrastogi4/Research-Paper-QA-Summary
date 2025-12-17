from fastapi import APIRouter, HTTPException, status

from api.schemas.health import HealthCheckResponse
# from core.health import health_checker, ServiceStatus
from core.logging import get_logger

logger = get_logger()
router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/")
async def health_check():
    """
    Check the health of the application.
    """
    return {"status": "health is ok"}
    # try:
    #     health_status = await health_checker.check_all_services()

    #     if health_status["status"] == ServiceStatus.HEALTHY:
    #         status_code = status.HTTP_200_OK
    #     elif health_status["status"] == ServiceStatus.DEGRADED:
    #         status_code = status.HTTP_206_PARTIAL_CONTENT
    #     else:
    #         status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    #     return HealthCheckResponse(**health_status)
    # except Exception as e:
    #     logger.error(f"Health check failed: {e}")
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/render-health")
async def render_health_check():
    return {"status": "ok"}
