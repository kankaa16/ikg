
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from DataProcess_fast import load_data
from ADEA import ADEA
from modelUtil import get_train_set, align_loss


class Args:
    pass


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataDir",
        default="./dataSet/d_w_15k"
    )

    parser.add_argument(
        "--epoach",
        type=int,
        default=5
    )

    parser.add_argument(
        "--save",
        default="./caea_demo"
    )

    args_cli = parser.parse_args()

    # --------------------------------------------------
    # CAEA arguments
    # --------------------------------------------------

    args = Args()

    args.dataDir = args_cli.dataDir
    args.input_dim = 150
    args.output_dim = 150
    args.rel_dim = 100
    args.attr_dim = 100
    args.num_layers = 2
    args.head = 1
    args.epoach = args_cli.epoach
    args.dropout_rate = 0.3
    args.lr = 0.001
    args.maxAttrNum = 20

    print("\n========================================")
    print("        CAEA FAST TRAINING")
    print("========================================\n")

    # --------------------------------------------------
    # Load CAEA data
    # --------------------------------------------------

    np.random.seed(42)

    train_pair, dev_pair, test_pair, data, AVH = load_data(
        args.dataDir,
        train_ratio=0.3,
        dev_ratio=0.7,
        test_ratio=0
    )

    node_size = len(data["ent_data"]["id_ent"])
    rel_size = len(data["rel_data"]["id_rel"]) * 2 - 1
    attr_size = len(data["attr_data"]["id_attr"])

    print("\nEntities:", node_size)
    print("Relations:", rel_size)
    print("Attributes:", attr_size)
    print("Training pairs:", len(train_pair))
    print("Validation pairs:", len(dev_pair))

    # --------------------------------------------------
    # Correct CAEA matrices
    # --------------------------------------------------

    matrix = data["all_matrix"]

    index_matrix = matrix["index_bi"]
    all_matrix = matrix["addself_bi"]

    attr_matrix = data["attr_matrix"]

    print("\nGraph matrices ready.")
    print("index_matrix shape:", index_matrix.shape)
    print("all_matrix shape:", all_matrix.shape)
    print("attr_matrix shape:", attr_matrix.shape)

    # --------------------------------------------------
    # Build ADEA model
    # --------------------------------------------------

    print("\nBuilding ADEA model...")

    model = ADEA(
        args,
        node_size,
        rel_size,
        attr_size,
        index_matrix=index_matrix,
        all_matix=all_matrix,
        attr_matrix=attr_matrix
    )

    # Build model once before training
    model(
        AVH=AVH,
        index_matrix=index_matrix,
        all_matix=all_matrix,
        attr_matrix=attr_matrix,
        training=False
    )

    print("Model built successfully.")

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = tf.keras.optimizers.legacy.RMSprop(
        learning_rate=args.lr
    )

    batch_size = node_size

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\nStarting training...\n")

    for turn in range(5):

        print(
            "\n========== TURN {}/5 ==========\n"
            .format(turn + 1)
        )

        for epoch in range(args.epoach):

            with tf.GradientTape() as tape:

                train_set = get_train_set(
                    node_size,
                    train_pair
                )

                ent_emb, conc_emb = model(
                    AVH=AVH,
                    index_matrix=index_matrix,
                    all_matix=all_matrix,
                    attr_matrix=attr_matrix,
                    training=True
                )

                loss = align_loss(
                    ent_emb,
                    train_set
                ) / batch_size

            grads = tape.gradient(
                loss,
                model.trainable_variables
            )

            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            print(
                "Turn {}/5 | Epoch {}/{} | Loss = {:.6f}".format(
                    turn + 1,
                    epoch + 1,
                    args.epoach,
                    float(loss)
                )
            )

    # --------------------------------------------------
    # Get final embeddings
    # --------------------------------------------------

    print("\nGenerating final entity embeddings...")

    ent_emb, conc_emb = model(
        AVH=AVH,
        index_matrix=index_matrix,
        all_matix=all_matrix,
        attr_matrix=attr_matrix,
        training=False
    )

    ent_emb = ent_emb.numpy()

    print("Entity embedding shape:", ent_emb.shape)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    save_dir = Path(args_cli.save)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    weights_path = save_dir / "caea.weights.h5"
    embeddings_path = save_dir / "entity_embeddings.npy"

    model.save_weights(
        str(weights_path)
    )

    np.save(
        embeddings_path,
        ent_emb
    )

    print("\n========================================")
    print("          TRAINING COMPLETE")
    print("========================================")

    print("\nSaved:")
    print(weights_path)
    print(embeddings_path)

    print("\nYou can now use the embeddings for")
    print("fast entity matching without retraining.")


if __name__ == "__main__":
    main()