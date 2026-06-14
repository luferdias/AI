"""
API de Eficiência Energética em Edifício Público
=================================================

Trabalho final da disciplina "Construção de APIs para Inteligência Artificial" (UFG).

Disponibiliza dois serviços de IA com operações distintas:
    1. Classificação da classe energética A–E (KNN, ML supervisionado);
    2. Resumo executivo do laudo (LLM com fallback determinístico).

Endpoints de apoio: cadastro/consulta de faturas, análise consolidada de consumo
e diagnóstico de demanda (regra). Os serviços de IA e o diagnóstico exigem
autenticação JWT (Bearer token obtido em /token).
"""

import logging

from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from routers.auth_router import get_current_active_user, router as auth_router
from routers.diagnostico_router import router as diagnostico_router
from routers.faturas_router import router as faturas_router
from routers.health_router import router as health_router
from routers.ia_router import router as ia_router

load_dotenv(find_dotenv())
logger = logging.getLogger("eficiencia-energetica")

app = FastAPI(
    title="API de Eficiência Energética em Edifício Público",
    description=(
        "Trabalho final — Construção de APIs para IA (UFG). Dois serviços de IA: "
        "**classificação** da classe energética (A–E) via KNN e **resumo executivo** "
        "do laudo via LLM. Inclui faturas, análise consolidada e diagnóstico de demanda."
    ),
    version="1.0.0",
    contact={"name": "Luís Fernando Dias", "email": "luis.dias@example.gov.br"},
    license_info={"name": "MIT"},
)


# --------------------------------------------------------------------------- #
# Tratamento padronizado de erros
# --------------------------------------------------------------------------- #

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Erro de validação em {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"erro": "Dados de entrada inválidos", "detalhes": exc.errors()},
    )


# --------------------------------------------------------------------------- #
# Registro das rotas
# --------------------------------------------------------------------------- #

app.include_router(health_router, tags=["Health Check"])
app.include_router(auth_router, tags=["Autenticação"])
app.include_router(faturas_router, tags=["Faturas e Análises"])

# Endpoints sensíveis protegidos por JWT.
app.include_router(
    diagnostico_router, tags=["Diagnóstico de Demanda"],
    dependencies=[Depends(get_current_active_user)],
)
app.include_router(
    ia_router, tags=["IA"],
    dependencies=[Depends(get_current_active_user)],
)


@app.get("/", tags=["Health Check"], summary="Raiz da API")
def raiz() -> dict:
    return {
        "projeto": "API de Eficiência Energética em Edifício Público",
        "servicos_ia": ["/api/v1/ia/classificar", "/api/v1/ia/resumir-analise"],
        "documentacao": "/docs",
    }
