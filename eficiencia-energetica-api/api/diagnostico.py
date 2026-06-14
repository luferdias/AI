"""
Núcleo do diagnóstico de demanda (Grupo A) — regra determinística DR vs DC.

Tolerância de ±5% sobre a demanda contratada:
    DR < 0,95·DC          -> demanda não utilizada (capacidade ociosa paga)
    DR > 1,05·DC          -> demanda ultrapassada (excedente faturável)
    0,95·DC <= DR <= 1,05·DC -> demanda eficiente (faixa ótima de aderência)

Observação: demanda é potência (kW). O valor monetário decorre de multiplicar
o diferencial (kW) pela tarifa de demanda (R$/kW).
"""

from typing import Optional

from models import SituacaoDemanda


def diagnosticar(dr: float, dc: float, tarifa_demanda: Optional[float] = None) -> dict:
    limite_inferior = 0.95 * dc
    limite_superior = 1.05 * dc

    if dr < limite_inferior:
        situacao = SituacaoDemanda.nao_utilizada
        diferenca = dc - dr
        narrativa = (
            f"A edificação mantém contratada demanda de {dc:.2f} kW, mas registrou apenas "
            f"{dr:.2f} kW — abaixo do piso de {limite_inferior:.2f} kW. Há {diferenca:.2f} kW "
            f"de capacidade reservada e paga, porém não utilizada, representando recurso "
            f"público imobilizado. Recomenda-se reavaliar a demanda contratada junto à "
            f"distribuidora, ajustando-a ao perfil real de carga."
        )
    elif dr > limite_superior:
        situacao = SituacaoDemanda.ultrapassada
        diferenca = dr - dc
        narrativa = (
            f"A edificação registrou {dr:.2f} kW, superando em {diferenca:.2f} kW a demanda "
            f"contratada de {dc:.2f} kW e rompendo o teto de tolerância de {limite_superior:.2f} kW. "
            f"Configura-se ultrapassagem de demanda, sujeita a faturamento adicional. Recomenda-se "
            f"gerir a curva de carga para aplainar picos e, se o excedente for estrutural, rever a "
            f"demanda contratada."
        )
    else:
        situacao = SituacaoDemanda.eficiente
        diferenca = 0.0
        narrativa = (
            f"A edificação registrou {dr:.2f} kW, dentro da faixa de eficiência de "
            f"{limite_inferior:.2f} a {limite_superior:.2f} kW (±5% sobre {dc:.2f} kW). A potência "
            f"contratada está aderente ao consumo real: sem capacidade ociosa relevante nem risco "
            f"de ultrapassagem. Recomenda-se manter o monitoramento contínuo."
        )

    valor_financeiro = round(diferenca * tarifa_demanda, 2) if tarifa_demanda is not None else None

    return {
        "situacao": situacao,
        "dr": dr,
        "dc": dc,
        "limite_inferior": round(limite_inferior, 4),
        "limite_superior": round(limite_superior, 4),
        "diferenca_kw": round(diferenca, 4),
        "valor_financeiro": valor_financeiro,
        "narrativa": narrativa,
    }
