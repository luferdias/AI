"""
Treino do classificador de classe energética (A–E) com KNN.

IMPORTANTE — natureza do dataset de treino
==========================================
A planilha de faturas não traz rótulos de classe energética (A–E). Para treinar
um classificador supervisionado, este script GERA um conjunto de treino SINTÉTICO,
porém calibrado a faixas plausíveis para edifícios públicos de escritório no clima
de Vitória-ES, e atribui o rótulo A–E por uma RUBRICA HEURÍSTICA TRANSPARENTE
(função `rotular`). Não se trata de valores normativos oficiais do PBE Edifica/INI-C,
e sim de uma heurística pedagógica documentada, adequada ao escopo do trabalho.
Em produção, o dataset deveria ser substituído por edifícios reais já etiquetados.

Reprodutibilidade: semente fixa (SEED). Rode `python ml/treinar_modelo.py` a partir
da pasta `api/` para (re)gerar `ml/modelo_knn.joblib`.
"""

import os

import numpy as np
from joblib import dump
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

SEED = 42
N_AMOSTRAS = 1200
CAMINHO_MODELO = os.path.join(os.path.dirname(__file__), "modelo_knn.joblib")

# Ordem das features — DEVE coincidir com VetorFeatures.ORDEM em models.py.
FEATURES = ["consumo_especifico", "fator_carga", "dpi",
            "densidade_climatizacao", "fracao_fv", "densidade_ocupacao"]


def rotular(consumo_especifico, fator_carga, dpi, densidade_climatizacao,
            fracao_fv, densidade_ocupacao) -> str:
    """Rubrica heurística -> escore de eficiência -> classe A–E.

    Cada eixo contribui com um escore parcial (quanto maior, mais eficiente).
    Pesos refletem a relevância típica de cada sistema no consumo predial.
    """
    # Normalizações simples para faixas plausíveis (0 = pior, 1 = melhor).
    s_consumo = np.clip((250 - consumo_especifico) / (250 - 80), 0, 1)        # menor é melhor
    s_carga = np.clip((fator_carga - 0.25) / (0.65 - 0.25), 0, 1)             # maior é melhor
    s_dpi = np.clip((16 - dpi) / (16 - 7), 0, 1)                              # menor é melhor
    s_clima = np.clip((60 - densidade_climatizacao) / (60 - 25), 0, 1)        # menor é melhor
    s_fv = np.clip(fracao_fv / 0.5, 0, 1)                                     # maior é melhor

    escore = (0.34 * s_consumo + 0.16 * s_carga + 0.16 * s_dpi
              + 0.16 * s_clima + 0.18 * s_fv)

    if escore >= 0.80:
        return "A"
    if escore >= 0.62:
        return "B"
    if escore >= 0.44:
        return "C"
    if escore >= 0.26:
        return "D"
    return "E"


def gerar_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    consumo_especifico = rng.uniform(80, 260, n)
    fator_carga = rng.uniform(0.22, 0.68, n)
    dpi = rng.uniform(6.5, 17, n)
    densidade_climatizacao = rng.uniform(22, 65, n)
    fracao_fv = rng.uniform(0.0, 0.55, n)
    densidade_ocupacao = rng.uniform(0.04, 0.18, n)

    X = np.column_stack([consumo_especifico, fator_carga, dpi,
                         densidade_climatizacao, fracao_fv, densidade_ocupacao])
    y = np.array([rotular(*linha) for linha in X])
    return X, y


def treinar() -> Pipeline:
    X, y = gerar_dataset(N_AMOSTRAS, SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])
    grade = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],  # Manhattan e Euclidiana
    }
    busca = GridSearchCV(pipe, grade, cv=5, scoring="accuracy", n_jobs=-1)
    busca.fit(X_tr, y_tr)

    print("Melhores hiperparâmetros:", busca.best_params_)
    print(f"Acurácia (validação cruzada): {busca.best_score_:.3f}")
    print("\nRelatório no conjunto de teste:")
    print(classification_report(y_te, busca.predict(X_te)))

    dump(busca.best_estimator_, CAMINHO_MODELO)
    print(f"Modelo persistido em: {CAMINHO_MODELO}")
    return busca.best_estimator_


if __name__ == "__main__":
    treinar()
