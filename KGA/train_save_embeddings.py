import argparse
import numpy as np
import tensorflow as tf

from DataProcess import load_data
from ADEA import ADEA
from modelUtil import get_train_set, align_loss


class Args:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", default="./dataSet/d_w_15k")
    parser.add_argument("--epoach", type=int, default=5)
    parser.add_argument("--save", default="./caea_demo")
    args_cli = parser.parse_args()

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
    print("       CAEA TRAIN + SAVE DEMO")
    print("========================================\n")

    train_pair, dev_pair, test_pair, data, AVH = load_data(
        args.dataDir, train_ratio=0.3, dev_ratio=0.7, test_ratio=0
    )

    node_size = len(data["ent_data"]["id_ent"])
    rel_size = len(data["rel_data"]["id_rel"]) * 2 - 1
    attr_size = len(data["attr_data"]["id_attr"])

    matrix = data["all_matrix"]
    index_matrix = matrix["index_bi"]
    all_matrix = matrix["addself_bi"]
    attr_matrix = data["attr_matrix"]

    model = ADEA(
        args, node_size, rel_size, attr_size,
        index_matrix=index_matrix,
        all_matix=all_matrix,
        attr_matrix=attr_matrix
    )

    # Build the model with the same inputs used by the original training code.
    model(
        AVH=AVH,
        index_matrix=index_matrix,
        all_matix=all_matrix,
        attr_matrix=attr_matrix,
        training=False
    )

    optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=args.lr)
    batch_size = node_size

    print("\nTraining and saving the trained CAEA model...\n")

    # Keep the repo's 5 outer turns, but expose the inner epoch count.
    # This mirrors the structure in modelUtil.py.
    for turn in range(5):
        for i in range(args.epoach):
            with tf.GradientTape() as tape:
                train_set = get_train_set(node_size, train_pair)

                ent_emb, conc_emb = model(
                    AVH=AVH,
                    index_matrix=index_matrix,
                    all_matix=all_matrix,
                    attr_matrix=attr_matrix,
                    training=True
                )

                loss = align_loss(ent_emb, train_set) / batch_size

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            print(
                "Turn {:d}/5 | Epoch {:d}/{:d} | Loss = {:.6f}".format(
                    turn + 1, i + 1, args.epoach, float(loss)
                )
            )

    save_dir = Path(args_cli.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save TensorFlow/Keras weights for the full model.
    weights_path = save_dir / "caea.weights.h5"
    model.save_weights(str(weights_path))

    # Generate the final embeddings once and save them.
    ent_emb, conc_emb = model(
        AVH=AVH,
        index_matrix=index_matrix,
        all_matix=all_matrix,
        attr_matrix=attr_matrix,
        training=False
    )

    np.save(save_dir / "entity_embeddings.npy", ent_emb.numpy())

    print("\n========================================")
    print("SAVED")
    print("========================================")
    print("Weights     :", weights_path)
    print("Embeddings  :", save_dir / "entity_embeddings.npy")
    print("\nNow you do NOT need to train again for the matching demo.")
    print("Run:")
    print("    python demo_match.py 5841")
    print()


if __name__ == "__main__":
    main()
