"""
Inferência do classificador de classe energética (A–E).

Carrega o modelo KNN persistido uma única vez (na importação) e expõe a função
pura `classificar_vetor`, separada da camada web para permitir teste isolado.
"""

import os

from joblib import load

_CAMINHO = os.path.join(os.path.dirname(__file__), "modelo_knn.joblib")

if not os.path.exists(_CAMINHO):
    raise FileNotFoundError(
        f"Modelo não encontrado em {_CAMINHO}. Execute 'python ml/treinar_modelo.py' "
        f"a partir da pasta api/ para gerá-lo."
    )

_modelo = load(_CAMINHO)


def classificar_vetor(vetor: list[float]) -> dict:
    """Classifica um vetor de features na classe energética A–E.

    Args:
        vetor: features na ordem fixa de VetorFeatures.ORDEM.

    Returns:
        dict com classe predita, confiança (probabilidade da classe) e a
        distribuição completa de probabilidades.
    """
    classe = _modelo.predict([vetor])[0]
    probabilidades = _modelo.predict_proba([vetor])[0]
    classes = _modelo.classes_
    dist = {c: round(float(p), 4) for c, p in zip(classes, probabilidades)}
    confianca = round(float(max(probabilidades)), 4)
    return {"classe": classe, "confianca": confianca, "probabilidades": dist}
