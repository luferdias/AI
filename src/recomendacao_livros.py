"""Treinamento de recomendação de livros com embeddings (arquitetura Aula 22).

Exemplo de uso:
  python src/recomendacao_livros.py --user-id 276729 --top-n 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


def load_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required_cols = {"ISBN", "Titulo", "ID_usuario", "Notas"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    return data


def prepare_mappings(data: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    data = data.copy()
    data["ID_usuario"] = pd.Categorical(data["ID_usuario"])
    data["new_user_id"] = data["ID_usuario"].cat.codes

    data["ISBN"] = pd.Categorical(data["ISBN"])
    data["new_book_id"] = data["ISBN"].cat.codes

    num_users = data["new_user_id"].nunique()
    num_items = data["new_book_id"].nunique()

    return data, num_users, num_items


def build_model(num_users: int, num_items: int, embedding_dim: int) -> Model:
    user_input = Input(shape=(1,), name="user_id")
    user_embedding = Embedding(num_users, embedding_dim, name="user_embedding")(user_input)
    user_embedding = Flatten()(user_embedding)

    item_input = Input(shape=(1,), name="book_id")
    item_embedding = Embedding(num_items, embedding_dim, name="book_embedding")(item_input)
    item_embedding = Flatten()(item_embedding)

    concat = Concatenate()([user_embedding, item_embedding])
    x = Dense(1024, activation="relu")(concat)
    output = Dense(1, name="rating")(x)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(loss="mse", optimizer=SGD(learning_rate=0.08, momentum=0.9))
    return model


def train_model(
    model: Model,
    data: pd.DataFrame,
    epochs: int,
    batch_size: int,
    val_split: float,
) -> tuple[tf.keras.callbacks.History, float, tuple[np.ndarray, ...]]:
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(data))
    user_ids = data["new_user_id"].to_numpy(dtype=np.int32)[indices]
    book_ids = data["new_book_id"].to_numpy(dtype=np.int32)[indices]
    ratings = data["Notas"].to_numpy(dtype=np.float32)[indices]

    train_size = int((1 - val_split) * len(ratings))
    train_user = user_ids[:train_size]
    train_book = book_ids[:train_size]
    train_ratings = ratings[:train_size]

    val_user = user_ids[train_size:]
    val_book = book_ids[train_size:]
    val_ratings = ratings[train_size:]

    avg_rating = float(train_ratings.mean())
    train_ratings_centered = train_ratings - avg_rating
    val_ratings_centered = val_ratings - avg_rating

    history = model.fit(
        x=[train_user, train_book],
        y=train_ratings_centered,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_data=([val_user, val_book], val_ratings_centered),
    )

    return history, avg_rating, (val_user, val_book, val_ratings_centered)


def plot_loss(history: tf.keras.callbacks.History, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Perda (Treino)")
    plt.plot(history.history["val_loss"], label="Perda (Validação)")
    plt.title("Avaliação do Modelo - Função de Perda (MSE)")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Quadrático Médio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def recommend_for_user(
    model: Model,
    data: pd.DataFrame,
    user_id: int,
    avg_rating: float,
    top_n: int,
    predict_batch_size: int,
) -> pd.DataFrame:
    user_rows = data.loc[data["ID_usuario"] == user_id]
    if user_rows.empty:
        raise ValueError("ID_usuario não encontrado no dataset.")

    user_code = int(user_rows["new_user_id"].iloc[0])

    books_unique = data[["new_book_id", "Titulo", "ISBN"]].drop_duplicates()
    rated_books = set(user_rows["new_book_id"].unique())
    book_ids = books_unique["new_book_id"].to_numpy(dtype=np.int32)
    candidate_mask = ~np.isin(book_ids, list(rated_books))
    candidate_book_ids = book_ids[candidate_mask]
    candidate_books = books_unique.loc[candidate_mask].reset_index(drop=True)

    if len(candidate_book_ids) == 0:
        raise ValueError("Usuário já avaliou todos os livros disponíveis.")

    preds_parts: list[np.ndarray] = []
    for start in range(0, len(candidate_book_ids), predict_batch_size):
        batch_ids = candidate_book_ids[start : start + predict_batch_size]
        user_array = np.full(batch_ids.shape, user_code, dtype=np.int32)
        batch_preds = model.predict([user_array, batch_ids], verbose=0).reshape(-1)
        preds_parts.append(batch_preds)

    preds = np.concatenate(preds_parts) + avg_rating
    top_k = min(top_n, len(preds))
    top_indices = np.argpartition(-preds, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-preds[top_indices])]

    top_recs = candidate_books.iloc[top_indices].assign(score=preds[top_indices])
    return top_recs[["ISBN", "Titulo", "score"]].rename(columns={"score": "Score"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Recomendação de livros com embeddings.")
    parser.add_argument("--csv", default="Base_livros.csv", help="Caminho para o CSV")
    parser.add_argument("--user-id", type=int, required=True, help="ID_usuario para recomendar")
    parser.add_argument("--top-n", type=int, default=5, help="Quantidade de recomendações")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas")
    parser.add_argument("--embedding-dim", type=int, default=10, help="Dimensão do embedding")
    parser.add_argument("--batch-size", type=int, default=1024, help="Tamanho do batch")
    parser.add_argument("--val-split", type=float, default=0.2, help="Percentual de validação")
    parser.add_argument("--plot", default="loss.png", help="Arquivo de saída do gráfico")
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=4096,
        help="Tamanho do batch para predições",
    )

    args = parser.parse_args()

    data = load_data(Path(args.csv))
    data, num_users, num_items = prepare_mappings(data)

    model = build_model(num_users=num_users, num_items=num_items, embedding_dim=args.embedding_dim)

    history, avg_rating, _ = train_model(
        model,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    plot_loss(history, Path(args.plot))

    recommendations = recommend_for_user(
        model,
        data,
        user_id=args.user_id,
        avg_rating=avg_rating,
        top_n=args.top_n,
        predict_batch_size=args.predict_batch_size,
    )

    print("\nTop recomendações:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
