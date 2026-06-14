"""
Endpoint de diagnóstico de demanda (regra determinística DR vs DC).

POST /api/v1/diagnostico/demanda
"""

from fastapi import APIRouter

from diagnostico import diagnosticar
from models import DemandaEntrada, DemandaSaida
from utils import get_logger

logger = get_logger()
router = APIRouter(prefix="/api/v1")


@router.post("/diagnostico/demanda", response_model=DemandaSaida,
             summary="Diagnostica a aderência entre demanda registrada e contratada")
def diagnostico_demanda(entrada: DemandaEntrada) -> DemandaSaida:
    logger.info(f"Diagnóstico de demanda: DR={entrada.dr} DC={entrada.dc}")
    resultado = diagnosticar(entrada.dr, entrada.dc, entrada.tarifa_demanda)
    return DemandaSaida(**resultado)
