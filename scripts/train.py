import sys
from pathlib import Path

# Add project root to Python module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import SDBDataset
from src.models.cnn import SDBCNN
from src.training.losses import StableHeteroscedasticLoss
from src.training.trainer import Trainer


def load_config(config_path):
    """Load project configuration from YAML."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    return config


def create_train_val_split(
    dataset_size,
    validation_ratio=0.2,
    random_seed=42,
):
    """Create a random train/validation split."""

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    indices = np.arange(dataset_size)

    rng = np.random.default_rng(
        random_seed
    )

    rng.shuffle(indices)

    val_size = int(
        dataset_size * validation_ratio
    )

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return (
        train_indices.tolist(),
        val_indices.tolist()
    )


def main():

    # ==================================================
    # 1. Load configuration
    # ==================================================

    config = load_config(
        "configs/config.yaml"
    )

    print("=" * 60)
    print("SDB-PRO Training")
    print("=" * 60)

    print(
        f"Project: "
        f"{config['project']['name']}"
    )

    print(
        f"Version: "
        f"{config['project']['version']}"
    )

    # ==================================================
    # 2. Read configuration
    # ==================================================

    sample_path = (
        "outputs/samples/guam_samples.npz"
    )

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    validation_ratio = (
        config["sample"]["validation_ratio"]
    )

    random_seed = (
        config["sample"]["random_seed"]
    )

    device = config["training"]["device"]

    checkpoint_dir = (
        config["output"]["checkpoint_dir"]
    )

    # ==================================================
    # 3. Set random seed
    # ==================================================

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            random_seed
        )

    # ==================================================
    # 4. Load dataset
    # ==================================================

    dataset = SDBDataset(
        sample_path
    )

    print(
        f"\nTotal samples: "
        f"{len(dataset)}"
    )

    # ==================================================
    # 5. Train / validation split
    # ==================================================

    train_indices, val_indices = (
        create_train_val_split(
            dataset_size=len(dataset),
            validation_ratio=validation_ratio,
            random_seed=random_seed,
        )
    )

    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # ==================================================
    # 6. Create DataLoaders
    # ==================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Batch size: {batch_size}"
    )

    # ==================================================
    # 7. Create model
    # ==================================================

    model = SDBCNN(
        in_channels=config["model"]["in_channels"],
        patch_size=config["model"]["patch_size"],
    )

    print(
        f"\nModel: "
        f"{config['model']['name']}"
    )

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ==================================================
    # 8. Create loss
    # ==================================================

    criterion = StableHeteroscedasticLoss()

    # ==================================================
    # 9. Create optimizer
    # ==================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print(
        f"Learning rate: "
        f"{learning_rate}"
    )

    # ==================================================
    # 10. Create trainer
    # ==================================================

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # ==================================================
    # 11. Train
    # ==================================================

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    # ==================================================
    # 12. Save training history
    # ==================================================

    history_path = Path(
        "outputs/metrics/training_history.npz"
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        history_path,
        train_loss=np.array(
            history["train_loss"]
        ),
        val_loss=np.array(
            history["val_loss"]
        ),
    )

    print(
        f"\nTraining history saved to: "
        f"{history_path}"
    )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()