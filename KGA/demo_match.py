import argparse
import numpy as np
from DataProcess import load_data


def cosine_similarity(query, matrix):
    query = query / (np.linalg.norm(query) + 1e-8)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix @ query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=int, help="KG1 entity ID")
    parser.add_argument("--dataDir", default="./dataSet/d_w_15k")
    parser.add_argument("--modelDir", default="./caea_demo")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    print("\n========================================")
    print("          CAEA ENTITY MATCHER")
    print("========================================\n")

    _, _, _, data, _ = load_data(args.dataDir)

    id_ent = data["ent_data"]["id_ent"]

    if args.entity not in id_ent:
        raise ValueError("Entity ID not found.")

    if args.entity >= 15000:
        raise ValueError("Give an entity ID from KG1 (0-14999).")

    embeddings = np.load(
        args.modelDir + "/entity_embeddings.npy"
    )

    # D-W-15K uses IDs 0..14999 for KG1 and
    # 15000..29999 for KG2.
    kg2_ids = list(range(15000, 30000))

    query = embeddings[args.entity]
    scores = cosine_similarity(
        query, embeddings[kg2_ids]
    )

    order = np.argsort(scores)[::-1][:args.top]

    print("INPUT ENTITY (KG1)")
    print("ID :", args.entity)
    print("URI:", id_ent[args.entity])

    print("\nTOP {} MATCHES IN KG2\n".format(args.top))

    for rank, pos in enumerate(order, 1):
        candidate = kg2_ids[pos]
        print(
            "{}. ID = {} | score = {:.4f}".format(
                rank, candidate, scores[pos]
            )
        )
        print("   ", id_ent[candidate])

    # Read the benchmark's ground-truth correspondence.
    ground_truth = None
    with open(args.dataDir + "/ref_ent_ids", "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split("\t")
            if int(a) == args.entity:
                ground_truth = int(b)
                break

    print("\n----------------------------------------")

    if ground_truth is not None:
        print("GROUND TRUTH")
        print("ID :", ground_truth)
        print("URI:", id_ent[ground_truth])

        ranked_ids = [kg2_ids[p] for p in order]

        if ground_truth in ranked_ids:
            rank = ranked_ids.index(ground_truth) + 1
            print(
                "\n✓ CORRECT ENTITY IS IN TOP {}".format(rank)
            )
        else:
            print("\n✗ Correct entity is outside the shown top matches.")
    else:
        print("No ground-truth pair found for this KG1 entity.")

    print("----------------------------------------\n")


if __name__ == "__main__":
    main()
