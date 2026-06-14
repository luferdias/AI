"""
Camada de features do classificador energético.

O fiscal informa DADOS BRUTOS coletáveis; o sistema DERIVA o vetor de seis
features que o KNN consome. Inclui salvaguarda física para o fator de carga.
"""

from models import FeaturesBrutas, VetorFeatures


def derivar_features(b: FeaturesBrutas) -> VetorFeatures:
    consumo_especifico = b.consumo_anual_kwh / b.area_util_m2

    demanda_media = b.consumo_anual_kwh / b.horas_operacao_ano
    fator_carga = demanda_media / b.demanda_max_kw
    if fator_carga > 1:
        # Salvaguarda física: demanda média não pode exceder a máxima registrada.
        raise ValueError(
            f"Fator de carga = {fator_carga:.2f} (>1), fisicamente impossível: a demanda "
            f"média ({demanda_media:.1f} kW) excede a demanda máxima registrada "
            f"({b.demanda_max_kw:.1f} kW). Reveja consumo, horas de operação e demanda."
        )

    dpi = b.potencia_iluminacao_kw * 1000 / b.area_util_m2
    densidade_climatizacao = b.potencia_climatizacao_kw * 1000 / b.area_util_m2
    fracao_fv = b.geracao_fv_anual_kwh / b.consumo_anual_kwh
    densidade_ocupacao = b.ocupantes / b.area_util_m2

    return VetorFeatures(
        consumo_especifico=round(consumo_especifico, 4),
        fator_carga=round(fator_carga, 4),
        dpi=round(dpi, 4),
        densidade_climatizacao=round(densidade_climatizacao, 4),
        fracao_fv=round(fracao_fv, 4),
        densidade_ocupacao=round(densidade_ocupacao, 4),
    )
