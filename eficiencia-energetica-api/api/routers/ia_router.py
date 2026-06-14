"""
Serviços de Inteligência Artificial da API (operações distintas).

Serviço 1 — POST /api/v1/ia/classificar
    Classificação supervisionada (KNN) da classe energética A–E a partir dos
    dados brutos da edificação. Modelo treinado com GridSearchCV e persistido.

Serviço 2 — POST /api/v1/ia/resumir-analise
    Geração de um resumo executivo do laudo via LLM (Groq). Caso a chave do LLM
    não esteja configurada, recorre a um fallback determinístico por regra,
    garantindo que a banca execute o endpoint sem dependências externas.
"""

from fastapi import APIRouter, HTTPException, status

from features import derivar_features
from ml.classificador import classificar_vetor
from models import (
    ClassificacaoSaida,
    FeaturesBrutas,
    ResumoEntrada,
    ResumoSaida,
)
from utils import LLMIndisponivelError, execute_prompt, get_logger

logger = get_logger()
router = APIRouter(prefix="/api/v1")


# --------------------------------------------------------------------------- #
# Serviço de IA 1 — Classificação (KNN)
# --------------------------------------------------------------------------- #

@router.post("/ia/classificar", response_model=ClassificacaoSaida,
             summary="[IA] Classifica a classe energética (A–E) via KNN")
def classificar(entrada: FeaturesBrutas) -> ClassificacaoSaida:
    logger.info("Requisição de classificação energética recebida")
    try:
        vetor = derivar_features(entrada)
    except ValueError as exc:
        # Salvaguarda física (ex.: fator de carga > 1) -> entrada inconsistente.
        logger.warning(f"Derivação de features rejeitada: {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    resultado = classificar_vetor(vetor.to_vector())
    logger.info(f"Classe predita: {resultado['classe']} (confiança {resultado['confianca']})")
    return ClassificacaoSaida(
        classe=resultado["classe"],
        confianca=resultado["confianca"],
        vetor_features=vetor,
        probabilidades=resultado["probabilidades"],
    )


# --------------------------------------------------------------------------- #
# Serviço de IA 2 — Resumo executivo (LLM com fallback por regra)
# --------------------------------------------------------------------------- #

def _resumo_por_regra(e: ResumoEntrada) -> str:
    fp_ok = e.fator_potencia >= 0.92
    return (
        f"Laudo sintético de eficiência energética. A edificação foi classificada na "
        f"classe {e.classe.value}, com consumo específico de {e.consumo_especifico:.1f} "
        f"kWh/m²·ano e fração de geração fotovoltaica de {e.fracao_fv*100:.0f}% do consumo. "
        f"A situação de demanda registrada é '{e.situacao_demanda.value.replace('_', ' ')}'. "
        f"O fator de potência médio de {e.fator_potencia:.3f} encontra-se "
        f"{'dentro' if fp_ok else 'abaixo'} do limite regulatório de 0,92, "
        f"{'não havendo' if fp_ok else 'havendo'} cobrança de energia reativa excedente. "
        f"Recomenda-se priorizar as medidas associadas à situação de demanda diagnosticada "
        f"e, conforme a classe obtida, avaliar intervenções em iluminação, climatização e "
        f"ampliação da geração fotovoltaica."
    )


@router.post("/ia/resumir-analise", response_model=ResumoSaida,
             summary="[IA] Gera resumo executivo do laudo (LLM com fallback por regra)")
def resumir_analise(entrada: ResumoEntrada) -> ResumoSaida:
    logger.info("Requisição de resumo executivo recebida")
    prompt = (
        "Você é um engenheiro especialista em eficiência energética de edifícios públicos. "
        "Redija um resumo executivo, técnico e objetivo (máximo 3 parágrafos, em português "
        "formal), a partir dos seguintes indicadores de uma edificação do Grupo A:\n"
        f"- Classe energética: {entrada.classe.value}\n"
        f"- Situação de demanda: {entrada.situacao_demanda.value}\n"
        f"- Consumo específico: {entrada.consumo_especifico:.1f} kWh/m²·ano\n"
        f"- Fração de geração fotovoltaica: {entrada.fracao_fv*100:.0f}%\n"
        f"- Fator de potência médio: {entrada.fator_potencia:.3f} (limite regulatório 0,92)\n"
        "Conclua com recomendações priorizadas de eficiência energética."
    )
    try:
        texto = execute_prompt(prompt)
        return ResumoSaida(resumo=texto, origem="llm")
    except LLMIndisponivelError:
        logger.info("LLM indisponível — gerando resumo por regra determinística")
        return ResumoSaida(resumo=_resumo_por_regra(entrada), origem="regra")
