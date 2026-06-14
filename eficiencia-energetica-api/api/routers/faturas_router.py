"""
Endpoints de faturas e análise consolidada de consumo.

POST /api/v1/faturas            -> registra dados mensais (validação Pydantic)
GET  /api/v1/faturas            -> lista faturas registradas
GET  /api/v1/faturas/{mes_ano}  -> consulta uma fatura (404 se inexistente)
GET  /api/v1/analises/consumo   -> indicadores consolidados do período
"""

from fastapi import APIRouter, HTTPException, status

from database import repositorio
from models import AnaliseConsumo, FaturaArmazenada, FaturaEntrada, SituacaoDemanda
from diagnostico import diagnosticar
from utils import get_logger

logger = get_logger()
router = APIRouter(prefix="/api/v1")


@router.post("/faturas", response_model=FaturaArmazenada,
             status_code=status.HTTP_201_CREATED, summary="Registra uma fatura mensal")
def criar_fatura(fatura: FaturaEntrada) -> FaturaArmazenada:
    if repositorio.obter_por_mes(fatura.mes_ano) is not None:
        logger.warning(f"Tentativa de duplicar fatura da competência {fatura.mes_ano}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Já existe fatura registrada para a competência {fatura.mes_ano}",
        )
    registro = repositorio.adicionar(fatura)
    logger.info(f"Fatura registrada: competência {registro.mes_ano}, id {registro.id}")
    return registro


@router.get("/faturas", response_model=list[FaturaArmazenada], summary="Lista as faturas")
def listar_faturas() -> list[FaturaArmazenada]:
    return repositorio.listar()


@router.get("/faturas/{mes_ano}", response_model=FaturaArmazenada,
            summary="Consulta uma fatura por competência")
def obter_fatura(mes_ano: str) -> FaturaArmazenada:
    fatura = repositorio.obter_por_mes(mes_ano)
    if fatura is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fatura da competência {mes_ano} não encontrada",
        )
    return fatura


@router.get("/analises/consumo", response_model=AnaliseConsumo,
            summary="Indicadores consolidados de consumo e demanda")
def analise_consumo() -> AnaliseConsumo:
    faturas = repositorio.listar()
    if not faturas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Não há faturas registradas para consolidar",
        )

    n = len(faturas)
    consumo_total = sum(f.consumo_total_kwh for f in faturas)
    geracao_total = sum(f.geracao_fv_kwh for f in faturas)

    ultrapassagem = sum(
        1 for f in faturas
        if diagnosticar(f.demanda_registrada_kw, f.demanda_contratada_kw)["situacao"]
        == SituacaoDemanda.ultrapassada
    )
    subutilizacao = sum(
        1 for f in faturas
        if diagnosticar(f.demanda_registrada_kw, f.demanda_contratada_kw)["situacao"]
        == SituacaoDemanda.nao_utilizada
    )

    logger.info(f"Análise consolidada gerada sobre {n} faturas")
    return AnaliseConsumo(
        total_faturas=n,
        periodo_inicio=faturas[0].mes_ano,
        periodo_fim=faturas[-1].mes_ano,
        consumo_total_kwh=round(consumo_total, 2),
        consumo_medio_mensal_kwh=round(consumo_total / n, 2),
        geracao_fv_total_kwh=round(geracao_total, 2),
        fracao_fv_media=round(geracao_total / consumo_total, 4) if consumo_total else 0.0,
        demanda_registrada_media_kw=round(sum(f.demanda_registrada_kw for f in faturas) / n, 2),
        fator_potencia_medio=round(sum(f.fator_potencia for f in faturas) / n, 4),
        meses_com_ultrapassagem=ultrapassagem,
        meses_subutilizacao=subutilizacao,
        valor_total_faturas=round(sum(f.valor_fatura for f in faturas), 2),
    )
