from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history_path,
    output_path,
):
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    history_path : str or Path
        Path to training_history.npz.

    output_path : str or Path
        Path to output PNG figure.
    """

    history_path = Path(history_path)
    output_path = Path(output_path)

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}"
        )

    data = np.load(history_path)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    if len(train_loss) != len(val_loss):
        raise ValueError(
            "Training and validation loss "
            "must have the same length."
        )

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SDB-PRO Training History")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Training curve saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":

    plot_training_history(
        history_path=(
            "outputs/metrics/"
            "training_history.npz"
        ),
        output_path=(
            "outputs/figures/"
            "training_history.png"
        ),
    )