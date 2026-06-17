"""Exercício complementar: treino e teste de classificador LSTM no dataset SLR.

Requisitos do enunciado:
- Texto em `title-abstract`
- Rótulo em `label`
- Embedding de 300 dimensões
- 2 camadas LSTM com 300 neurônios cada
- 1 camada Densa com 300 neurônios
- Saída com 1 neurônio
- Máximo de 200 tokens por sequência
- Treino por 10 épocas
- Avaliar acurácia, precisão, recall e F-score por classe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

DATA_URL = "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/slr.csv"
DATA_FILE = Path("slr.csv")
EPOCHS = 10
MAX_TOKENS = 200


def download_dataset_if_needed(file_path: Path = DATA_FILE) -> None:
    if file_path.exists():
        print(f"Arquivo {file_path} já existe. Tudo OK.")
        return

    df = pd.read_csv(DATA_URL)
    df.to_csv(file_path, index=False)
    print(f"Dataset baixado e salvo em {file_path}.")


def load_and_validate_data(file_path: Path) -> tuple[pd.Series, np.ndarray, LabelEncoder]:
    df = pd.read_csv(file_path)

    required = {"title-abstract", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    X = df["title-abstract"].fillna("").astype(str)
    y_raw = df["label"].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    if len(encoder.classes_) != 2:
        raise ValueError(
            "Este script espera classificação binária (saída com 1 neurônio). "
            f"Classes encontradas: {list(encoder.classes_)}"
        )

    return X, y, encoder


def prepare_sequences(
    X_train: pd.Series,
    X_test: pd.Series,
    *,
    max_tokens: int,
    max_words: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(train_seq, maxlen=max_tokens, padding="post", truncating="post")
    X_test_pad = pad_sequences(test_seq, maxlen=max_tokens, padding="post", truncating="post")

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    return X_train_pad, X_test_pad, vocab_size


def build_model(vocab_size: int, max_tokens: int) -> Sequential:
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=300, input_length=max_tokens),
            LSTM(300, return_sequences=True),
            LSTM(300),
            Dense(300, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento/teste de classificador LSTM no dataset SLR.")
    parser.add_argument("--data-file", type=Path, default=DATA_FILE, help="Arquivo local slr.csv")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fração de teste")
    parser.add_argument("--random-state", type=int, default=42, help="Seed aleatória")
    parser.add_argument("--max-words", type=int, default=50000, help="Máximo de palavras no vocabulário")
    args = parser.parse_args()

    tf.random.set_seed(args.random_state)
    np.random.seed(args.random_state)

    download_dataset_if_needed(args.data_file)
    X, y, encoder = load_and_validate_data(args.data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    X_train_pad, X_test_pad, vocab_size = prepare_sequences(
        X_train,
        X_test,
        max_tokens=MAX_TOKENS,
        max_words=args.max_words,
    )

    model = build_model(vocab_size=vocab_size, max_tokens=MAX_TOKENS)
    model.fit(X_train_pad, y_train, epochs=EPOCHS, batch_size=args.batch_size, verbose=2)

    y_proba = model.predict(X_test_pad, verbose=0).reshape(-1)
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    print("\nResultados no conjunto de teste")
    print(f"Acurácia geral: {acc:.4f}")
    print("\nMétricas por classe:")
    for idx, class_name in enumerate(encoder.classes_):
        print(
            f"- Classe '{class_name}': "
            f"Precision={precision[idx]:.4f}, "
            f"Recall={recall[idx]:.4f}, "
            f"F-Score={fscore[idx]:.4f}, "
            f"Support={support[idx]}"
        )


if __name__ == "__main__":
    main()
