import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


# ================================================================
# Project root
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.dataset import SDBDataset
from src.models.cnn import SDBCNN
from src.inference.predictor import SDBPredictor
from src.evaluation.metrics import calculate_all_metrics


# ================================================================
# Configuration
# ================================================================

def load_config(config_path):
    """
    Load YAML configuration file.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ================================================================
# Train / validation split
# ================================================================

def create_train_val_split(
    dataset_size,
    validation_ratio=0.2,
    random_seed=42
):
    """
    Reproduce the same train/validation split used during training.

    Returns
    -------
    train_indices : list
    val_indices : list
    """

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    indices = np.arange(dataset_size)

    rng = np.random.default_rng(random_seed)
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


# ================================================================
# Main evaluation
# ================================================================

def main():

    print("=" * 60)
    print("SDB-PRO Model Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------

    config = load_config(
        PROJECT_ROOT / "configs" / "config.yaml"
    )

    validation_ratio = config["sample"]["validation_ratio"]
    random_seed = config["sample"]["random_seed"]

    batch_size = config["training"]["batch_size"]

    in_channels = config["model"]["in_channels"]
    patch_size = config["model"]["patch_size"]

    device = config["training"]["device"]

    checkpoint_path = (
        PROJECT_ROOT
        / config["output"]["checkpoint_dir"]
        / "best_model.pth"
    )

    sample_path = (
        PROJECT_ROOT
        / "outputs"
        / "samples"
        / "guam_samples.npz"
    )

    metrics_dir = (
        PROJECT_ROOT
        / config["output"]["metrics_dir"]
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------

    dataset = SDBDataset(sample_path)

    print(
        f"\nTotal samples: "
        f"{len(dataset)}"
    )

    # ------------------------------------------------------------
    # 3. Reproduce validation split
    # ------------------------------------------------------------

    _, val_indices = create_train_val_split(
        dataset_size=len(dataset),
        validation_ratio=validation_ratio,
        random_seed=random_seed
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # ------------------------------------------------------------
    # 4. Collect reference depths
    # ------------------------------------------------------------

    y_true = np.concatenate(
        [
            y.numpy()
            for _, y in val_loader
        ]
    )

    print(
        f"Reference depth range: "
        f"{y_true.min():.3f} ~ "
        f"{y_true.max():.3f}"
    )

    # ------------------------------------------------------------
    # 5. Create model
    # ------------------------------------------------------------

    model = SDBCNN(
        in_channels=in_channels,
        patch_size=patch_size
    )

    print(
        f"\nModel: "
        f"{config['model']['name']}"
    )

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ------------------------------------------------------------
    # 6. Load checkpoint
    # ------------------------------------------------------------

    predictor = SDBPredictor(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # ------------------------------------------------------------
    # 7. Run prediction
    # ------------------------------------------------------------

    print("\nRunning validation inference...")

    y_pred, raw_log_variance = predictor.predict(
        val_loader
    )

    print(
        f"Prediction range: "
        f"{y_pred.min():.3f} ~ "
        f"{y_pred.max():.3f}"
    )

    # ------------------------------------------------------------
    # 8. Check sample alignment
    # ------------------------------------------------------------

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Reference and prediction arrays "
            "have different lengths."
        )

    # ------------------------------------------------------------
    # 9. Calculate metrics
    # ------------------------------------------------------------

    metrics = calculate_all_metrics(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("Validation Metrics")
    print("=" * 60)

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    print(
        f"Bias : {metrics['Bias']:.4f}"
    )

    # ------------------------------------------------------------
    # 10. Save metrics
    # ------------------------------------------------------------

    metrics_output = {
        "dataset": "Guam",
        "split": "validation",
        "total_samples": len(dataset),
        "validation_samples": len(val_dataset),
        "validation_ratio": validation_ratio,
        "random_seed": random_seed,
        "checkpoint_epoch": predictor.checkpoint.get(
            "epoch",
            None
        ),
        "checkpoint_val_loss": predictor.checkpoint.get(
            "val_loss",
            None
        ),
        "metrics": metrics
    }

    output_path = (
        metrics_dir
        / "validation_metrics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics_output,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nMetrics saved to: "
        f"{output_path}"
    )

    print("\nEvaluation completed.")


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()