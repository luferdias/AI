"""Treinamento de recomendação de livros com embeddings.

Exemplo de uso:
  python src/recomendacao_livros.py --user-id 276725 --top-n 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required_cols = {"ISBN", "Titulo", "ID_usuario", "Notas"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    return data


def prepare_mappings(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    user_codes, user_uniques = pd.factorize(data["ID_usuario"])
    item_codes, item_uniques = pd.factorize(data["ISBN"])

    data = data.copy()
    data["user_idx"] = user_codes
    data["item_idx"] = item_codes

    user_id_to_idx = {int(user_id): int(idx) for idx, user_id in enumerate(user_uniques)}
    item_id_to_idx = {int(item_id): int(idx) for idx, item_id in enumerate(item_uniques)}

    return data, user_id_to_idx, item_id_to_idx


def build_model(num_users: int, num_items: int, embedding_dim: int = 50) -> tf.keras.Model:
    user_input = tf.keras.layers.Input(shape=(1,), name="user_idx")
    item_input = tf.keras.layers.Input(shape=(1,), name="item_idx")

    user_embedding = tf.keras.layers.Embedding(
        input_dim=num_users,
        output_dim=embedding_dim,
        name="user_embedding",
    )(user_input)
    item_embedding = tf.keras.layers.Embedding(
        input_dim=num_items,
        output_dim=embedding_dim,
        name="item_embedding",
    )(item_input)

    dot = tf.keras.layers.Dot(axes=2)([user_embedding, item_embedding])
    dot = tf.keras.layers.Flatten()(dot)
    output = tf.keras.layers.Dense(1, activation="linear", name="rating")(dot)

    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_model(
    model: tf.keras.Model,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 256,
) -> tf.keras.callbacks.History:
    return model.fit(
        [train_data["user_idx"], train_data["item_idx"]],
        train_data["Notas"].astype("float32"),
        validation_data=(
            [val_data["user_idx"], val_data["item_idx"]],
            val_data["Notas"].astype("float32"),
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )


def plot_loss(history: tf.keras.callbacks.History, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title("Loss por época")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)



def recommend_for_user(
    model: tf.keras.Model,
    data: pd.DataFrame,
    user_id: int,
    user_id_to_idx: dict[int, int],
    item_id_to_idx: dict[int, int],
    top_n: int = 5,
) -> pd.DataFrame:
    if user_id not in user_id_to_idx:
        raise ValueError("ID_usuario não encontrado no dataset.")

    user_idx = user_id_to_idx[user_id]
    rated_items = set(data.loc[data["ID_usuario"] == user_id, "ISBN"].unique())

    all_items = np.array(list(item_id_to_idx.keys()))
    candidate_items = np.array([item for item in all_items if item not in rated_items])
    candidate_indices = np.array([item_id_to_idx[item] for item in candidate_items])

    user_array = np.full_like(candidate_indices, user_idx)
    preds = model.predict([user_array, candidate_indices], verbose=0).reshape(-1)

    top_indices = preds.argsort()[::-1][:top_n]
    top_items = candidate_items[top_indices]
    top_scores = preds[top_indices]

    titles = (
        data.drop_duplicates("ISBN")
        .set_index("ISBN")["Titulo"]
        .reindex(top_items)
        .fillna("Título não encontrado")
        .values
    )

    return pd.DataFrame(
        {
            "ISBN": top_items,
            "Titulo": titles,
            "Score": top_scores,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recomendação de livros com embeddings.")
    parser.add_argument("--csv", default="Base_livros.csv", help="Caminho para o CSV")
    parser.add_argument("--user-id", type=int, required=True, help="ID_usuario para recomendar")
    parser.add_argument("--top-n", type=int, default=5, help="Quantidade de recomendações")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Dimensão do embedding")
    parser.add_argument("--plot", default="loss.png", help="Arquivo de saída do gráfico")

    args = parser.parse_args()

    data = load_data(Path(args.csv))
    data, user_id_to_idx, item_id_to_idx = prepare_mappings(data)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    model = build_model(
        num_users=len(user_id_to_idx),
        num_items=len(item_id_to_idx),
        embedding_dim=args.embedding_dim,
    )

    history = train_model(model, train_data, val_data, epochs=args.epochs)
    plot_loss(history, Path(args.plot))

    recommendations = recommend_for_user(
        model,
        data,
        args.user_id,
        user_id_to_idx,
        item_id_to_idx,
        top_n=args.top_n,
    )

    print("\nTop recomendações:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
