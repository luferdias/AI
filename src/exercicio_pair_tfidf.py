"""Exercício: classificação de abstracts com TF-IDF e comparação de modelos.

Passos:
1) Baixa `pair.csv` caso não exista.
2) Extrai X da coluna `abstract` e y da coluna `label`.
3) Faz split treino/teste com test_size=0.3 e random_state=42.
4) Treina Decision Tree, Random Forest, Linear SVM e Regressão Logística.
5) Compara ROC-AUC e reporta o melhor classificador.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/pair.csv"
DATA_FILE = Path("pair.csv")


def download_dataset_if_needed(file_path: Path = DATA_FILE) -> None:
    if file_path.exists():
        print(f"Arquivo {file_path} já existe. Tudo OK.")
        return

    df = pd.read_csv(DATA_URL)
    df.to_csv(file_path, index=False)
    print(f"Dataset baixado e salvo em {file_path}.")


def _roc_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) == 2:
        return roc_auc_score(y_true, scores)
    return roc_auc_score(y_true, scores, multi_class="ovr", average="weighted")


def evaluate_models(df: pd.DataFrame) -> Dict[str, float]:
    if not {"abstract", "label"}.issubset(df.columns):
        raise ValueError("O dataset deve conter as colunas 'abstract' e 'label'.")

    X = df["abstract"].fillna("")
    y_raw = df["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Linear SVM": LinearSVC(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    }

    results: Dict[str, float] = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        model = pipeline.named_steps["model"]
        X_test_vec = pipeline.named_steps["tfidf"].transform(X_test)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_vec)
            scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
        else:
            decision = model.decision_function(X_test_vec)
            scores = decision[:, 1] if np.ndim(decision) == 2 and decision.shape[1] == 2 else decision

        auc = _roc_auc_from_scores(y_test, scores)
        results[name] = auc

    return results


def main() -> None:
    download_dataset_if_needed(DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    results = evaluate_models(df)

    print("\nROC-AUC por classificador:")
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"- {name}: {auc:.4f}")

    best_model = max(results, key=results.get)
    print(f"\nMelhor classificador (ROC-AUC): {best_model}")


if __name__ == "__main__":
    main()
