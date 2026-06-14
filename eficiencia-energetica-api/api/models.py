"""
Schemas Pydantic (contratos de dados) da API de Eficiência Energética.

Ancorados nas colunas reais do "Relatório de Análise de Eficiência Energética
do MGI/SRA-ES": faturas do Grupo A (demanda contratada/registrada, bandeira
tarifária, geração fotovoltaica, fator de potência).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums de domínio
# --------------------------------------------------------------------------- #

class Bandeira(str, Enum):
    verde = "verde"
    amarela = "amarela"
    vermelha = "vermelha"
    vermelha_2 = "vermelha_2"


class SituacaoDemanda(str, Enum):
    nao_utilizada = "demanda_nao_utilizada"
    ultrapassada = "demanda_ultrapassada"
    eficiente = "demanda_eficiente"


class ClasseEnergetica(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


# --------------------------------------------------------------------------- #
# Faturas de energia (entrada de dados mensais)
# --------------------------------------------------------------------------- #

class FaturaEntrada(BaseModel):
    mes_ano: str = Field(
        ..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
        description="Competência no formato AAAA-MM (ex.: 2025-03)", examples=["2025-03"],
    )
    consumo_total_kwh: float = Field(..., gt=0, description="Consumo ativo total no mês (kWh)")
    demanda_contratada_kw: float = Field(..., gt=0, description="Demanda contratada (kW)")
    demanda_registrada_kw: float = Field(..., gt=0, description="Demanda máxima registrada no mês (kW)")
    bandeira: Bandeira = Field(..., description="Bandeira tarifária vigente no mês")
    geracao_fv_kwh: float = Field(0.0, ge=0, description="Geração fotovoltaica no mês (kWh); 0 se não houver")
    fator_potencia: float = Field(..., gt=0, le=1, description="Fator de potência médio (0–1; referência ≥ 0,92)")
    valor_fatura: float = Field(..., gt=0, description="Valor total da fatura (R$)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "mes_ano": "2025-03",
                "consumo_total_kwh": 77579.37,
                "demanda_contratada_kw": 400.0,
                "demanda_registrada_kw": 371.52,
                "bandeira": "verde",
                "geracao_fv_kwh": 10080.0,
                "fator_potencia": 0.993,
                "valor_fatura": 57493.57,
            }
        }
    }


class FaturaArmazenada(FaturaEntrada):
    id: int = Field(..., description="Identificador sequencial atribuído pela API")


# --------------------------------------------------------------------------- #
# Análise consolidada de consumo
# --------------------------------------------------------------------------- #

class AnaliseConsumo(BaseModel):
    total_faturas: int
    periodo_inicio: Optional[str]
    periodo_fim: Optional[str]
    consumo_total_kwh: float
    consumo_medio_mensal_kwh: float
    geracao_fv_total_kwh: float
    fracao_fv_media: float = Field(..., description="Geração FV ÷ consumo total (adimensional)")
    demanda_registrada_media_kw: float
    fator_potencia_medio: float
    meses_com_ultrapassagem: int
    meses_subutilizacao: int
    valor_total_faturas: float


# --------------------------------------------------------------------------- #
# Diagnóstico de demanda (serviço por regra — DR vs DC)
# --------------------------------------------------------------------------- #

class DemandaEntrada(BaseModel):
    dr: float = Field(..., gt=0, description="Demanda registrada (kW)")
    dc: float = Field(..., gt=0, description="Demanda contratada (kW)")
    tarifa_demanda: Optional[float] = Field(None, gt=0, description="Tarifa de demanda (R$/kW), opcional")

    model_config = {"json_schema_extra": {"example": {"dr": 429.84, "dc": 400.0, "tarifa_demanda": 39.82}}}


class DemandaSaida(BaseModel):
    situacao: SituacaoDemanda
    dr: float
    dc: float
    limite_inferior: float
    limite_superior: float
    diferenca_kw: float
    valor_financeiro: Optional[float]
    narrativa: str


# --------------------------------------------------------------------------- #
# Serviço de IA 1 — Classificação da classe energética (KNN)
# --------------------------------------------------------------------------- #

class FeaturesBrutas(BaseModel):
    area_util_m2: float = Field(..., gt=0, description="Área útil/condicionada (m²)")
    consumo_anual_kwh: float = Field(
        ..., gt=0,
        description="Consumo TOTAL anual da edificação (rede + autoconsumo FV), em kWh",
    )
    demanda_max_kw: float = Field(..., gt=0, description="Demanda máxima registrada no período (kW)")
    horas_operacao_ano: float = Field(..., gt=0, le=8784, description="Horas de operação no ano (h)")
    potencia_iluminacao_kw: float = Field(..., gt=0, description="Potência instalada de iluminação (kW)")
    potencia_climatizacao_kw: float = Field(..., gt=0, description="Potência instalada de climatização (kW)")
    geracao_fv_anual_kwh: float = Field(..., ge=0, description="Geração fotovoltaica anual (kWh)")
    ocupantes: int = Field(..., gt=0, description="Número de ocupantes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "area_util_m2": 3000.0, "consumo_anual_kwh": 360000.0, "demanda_max_kw": 400.0,
                "horas_operacao_ano": 2900.0, "potencia_iluminacao_kw": 30.0,
                "potencia_climatizacao_kw": 120.0, "geracao_fv_anual_kwh": 90000.0, "ocupantes": 300,
            }
        }
    }


class VetorFeatures(BaseModel):
    consumo_especifico: float = Field(..., description="kWh/m²·ano")
    fator_carga: float = Field(..., description="adimensional (0–1)")
    dpi: float = Field(..., description="W/m²")
    densidade_climatizacao: float = Field(..., description="W/m²")
    fracao_fv: float = Field(..., description="adimensional")
    densidade_ocupacao: float = Field(..., description="pessoas/m²")

    # ORDEM IMUTÁVEL — contrato com o modelo treinado. Alterar exige retreinar o KNN.
    ORDEM: tuple = (
        "consumo_especifico", "fator_carga", "dpi",
        "densidade_climatizacao", "fracao_fv", "densidade_ocupacao",
    )

    def to_vector(self) -> list[float]:
        return [getattr(self, nome) for nome in self.ORDEM]


class ClassificacaoSaida(BaseModel):
    classe: ClasseEnergetica
    confianca: float = Field(..., ge=0, le=1, description="Probabilidade da classe predita")
    vetor_features: VetorFeatures
    probabilidades: dict = Field(..., description="Distribuição de probabilidade por classe")


# --------------------------------------------------------------------------- #
# Serviço de IA 2 — Resumo executivo do laudo (LLM, com fallback por regra)
# --------------------------------------------------------------------------- #

class ResumoEntrada(BaseModel):
    classe: ClasseEnergetica = Field(..., description="Classe energética obtida na classificação")
    situacao_demanda: SituacaoDemanda = Field(..., description="Situação de demanda diagnosticada")
    consumo_especifico: float = Field(..., gt=0, description="kWh/m²·ano")
    fracao_fv: float = Field(..., ge=0, description="Fração de geração fotovoltaica")
    fator_potencia: float = Field(..., gt=0, le=1, description="Fator de potência médio")

    model_config = {
        "json_schema_extra": {
            "example": {
                "classe": "B", "situacao_demanda": "demanda_eficiente",
                "consumo_especifico": 120.0, "fracao_fv": 0.25, "fator_potencia": 0.993,
            }
        }
    }


class ResumoSaida(BaseModel):
    resumo: str
    origem: str = Field(..., description="'llm' se gerado por modelo de linguagem; 'regra' se fallback determinístico")


class HealthCheck(BaseModel):
    status: str = "OK"
