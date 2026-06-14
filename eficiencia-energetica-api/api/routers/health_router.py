"""Health check para orquestração de contêineres e verificação de disponibilidade."""

from fastapi import APIRouter, status

from models import HealthCheck

router = APIRouter()


@router.get("/health", summary="Verificação de saúde da API",
            status_code=status.HTTP_200_OK, response_model=HealthCheck)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")
