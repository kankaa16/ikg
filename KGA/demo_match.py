
import argparse
import numpy as np
from pathlib import Path


def load_entities(path):
    entities = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                entity_id = int(parts[0])
                uri = parts[1]
                entities[entity_id] = uri

    return entities


def cosine_similarity(query, matrix):
    query = query / (np.linalg.norm(query) + 1e-8)

    matrix = matrix / (
        np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    )

    return matrix @ query


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "entity",
        type=int,
        help="KG1 entity ID"
    )

    parser.add_argument(
        "--dataDir",
        default="./dataSet/d_w_15k"
    )

    parser.add_argument(
        "--modelDir",
        default="/content/drive/MyDrive/CAEA/caea_demo_500"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5
    )

    args = parser.parse_args()

    data_dir = Path(args.dataDir)
    model_dir = Path(args.modelDir)

    # --------------------------------------------------
    # Load entity dictionaries
    # --------------------------------------------------

    entities_1 = load_entities(data_dir / "ent_ids_1")
    entities_2 = load_entities(data_dir / "ent_ids_2")

    # --------------------------------------------------
    # Load trained CAEA embeddings
    # --------------------------------------------------

    embeddings = np.load(
        model_dir / "entity_embeddings.npy"
    )

    print("\n========================================")
    print("       CAEA ENTITY MATCHING DEMO")
    print("========================================\n")

    print("Embedding shape:", embeddings.shape)

    # --------------------------------------------------
    # KG1 entity
    # --------------------------------------------------

    entity_id = args.entity

    if entity_id not in entities_1:
        print("Entity ID not found in KG1.")
        return

    source_name = entities_1[entity_id]

    print("KG1 entity:")
    print("ID :", entity_id)
    print("URI:", source_name)

    # --------------------------------------------------
    # Extract KG1 embedding
    # --------------------------------------------------

    query_embedding = embeddings[entity_id]

    # KG2 starts from ID 15000
    kg2_ids = sorted(entities_2.keys())

    kg2_embeddings = embeddings[kg2_ids]

    # --------------------------------------------------
    # Compare
    # --------------------------------------------------

    scores = cosine_similarity(
        query_embedding,
        kg2_embeddings
    )

    top_indices = np.argsort(scores)[::-1][:args.top]

    print("\nTop matching KG2 entities:\n")

    for rank, index in enumerate(top_indices, start=1):

        matched_id = kg2_ids[index]
        matched_uri = entities_2[matched_id]
        score = scores[index]

        print(
            "{}. ID={} | similarity={:.4f} | {}".format(
                rank,
                matched_id,
                score,
                matched_uri
            )
        )

    print("\n========================================")


if __name__ == "__main__":
    main()