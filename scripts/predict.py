import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml


# ================================================================
# Project root
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.cnn import SDBCNN
from src.inference.predictor import SDBPredictor


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
# Create sliding-window patches
# ================================================================

def create_patches(image, patch_size=5):
    """
    Create sliding-window patches from a raster.

    Parameters
    ----------
    image : np.ndarray
        Raster image with shape [C, H, W].

    patch_size : int
        Spatial size of the patch.

    Returns
    -------
    patches : np.ndarray
        Valid patch array with shape
        [N, C, patch_size, patch_size].

    valid_mask : np.ndarray
        Boolean mask with shape [H, W].

    valid_positions : np.ndarray
        Pixel coordinates of valid patch centers,
        shape [N, 2], stored as [row, col].
    """

    if image.ndim != 3:
        raise ValueError(
            f"image must have shape [C, H, W], "
            f"got {image.shape}"
        )

    if patch_size % 2 == 0:
        raise ValueError(
            "patch_size must be an odd number."
        )

    channels, height, width = image.shape

    half = patch_size // 2

    if height < patch_size or width < patch_size:
        raise ValueError(
            "Raster is smaller than the patch size."
        )

    # ------------------------------------------------------------
    # Create sliding windows
    #
    # [C, H, W]
    #      ↓
    # [C, H', W', patch, patch]
    # ------------------------------------------------------------

    windows = np.lib.stride_tricks.sliding_window_view(
        image,
        window_shape=(patch_size, patch_size),
        axis=(1, 2)
    )

    # ------------------------------------------------------------
    # Rearrange dimensions
    #
    # [C, H', W', patch, patch]
    #      ↓
    # [H', W', C, patch, patch]
    # ------------------------------------------------------------

    windows = windows.transpose(
        1, 2, 0, 3, 4
    )

    valid_height = height - patch_size + 1
    valid_width = width - patch_size + 1

    # ------------------------------------------------------------
    # Generate all patch center coordinates
    # ------------------------------------------------------------

    center_rows = np.arange(
        half,
        height - half
    )

    center_cols = np.arange(
        half,
        width - half
    )

    row_grid, col_grid = np.meshgrid(
        center_rows,
        center_cols,
        indexing="ij"
    )

    center_positions = np.column_stack(
        [
            row_grid.ravel(),
            col_grid.ravel()
        ]
    )

    # ------------------------------------------------------------
    # Reshape patches
    #
    # [H', W', C, patch, patch]
    #      ↓
    # [N, C, patch, patch]
    # ------------------------------------------------------------

    patches = windows.reshape(
        valid_height * valid_width,
        channels,
        patch_size,
        patch_size
    )

    # ------------------------------------------------------------
    # Keep only patches whose entire contents are finite
    # ------------------------------------------------------------

    finite_mask = np.isfinite(
        patches
    ).all(
        axis=(1, 2, 3)
    )

    patches = patches[
        finite_mask
    ]

    valid_positions = center_positions[
        finite_mask
    ]

    # ------------------------------------------------------------
    # Create full-image valid mask
    # ------------------------------------------------------------

    valid_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    if len(valid_positions) > 0:

        valid_mask[
            valid_positions[:, 0],
            valid_positions[:, 1]
        ] = True

    return (
        patches,
        valid_mask,
        valid_positions
    )
# ================================================================
# Full-image inference
# ================================================================

def predict_raster(
    predictor,
    image,
    patch_size=5,
    batch_size=512
):
    """
    Run CNN inference over an entire raster.

    Parameters
    ----------
    predictor : SDBPredictor
        Trained SDB predictor.

    image : np.ndarray
        Input raster [C, H, W].

    patch_size : int
        Patch size used by the CNN.

    batch_size : int
        Number of patches processed per batch.

    Returns
    -------
    prediction_map : np.ndarray
        Predicted depth map [H, W].

    variance_map : np.ndarray
        Predicted variance map [H, W].

    valid_mask : np.ndarray
        Valid prediction mask [H, W].
    """

    print("\nCreating sliding-window patches...")

    patches, valid_mask, valid_positions = create_patches(
        image=image,
        patch_size=patch_size
    )

    print(
        f"Valid patches: {len(patches):,}"
    )

    if len(patches) == 0:
        raise ValueError(
            "No valid patches were found."
        )

    # ------------------------------------------------------------
    # Prepare output arrays
    # ------------------------------------------------------------

    height = image.shape[1]
    width = image.shape[2]

    prediction_map = np.full(
        (height, width),
        -9999.0,
        dtype=np.float32
    )

    variance_map = np.full(
        (height, width),
        -9999.0,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------

    predictions = []
    raw_log_variances = []

    print("\nRunning CNN inference...")

    for start in range(
        0,
        len(patches),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(patches)
        )

        batch = patches[start:end]

        # Convert to Tensor
        X = torch.from_numpy(
            batch.astype(np.float32)
        ).to(
            predictor.device
        )

        with torch.no_grad():

            output = predictor.model(X)

            depth = output[:, 0]

            raw_log_var = output[:, 1]

        predictions.append(
            depth.cpu().numpy()
        )

        raw_log_variances.append(
            raw_log_var.cpu().numpy()
        )

        if (
            start == 0
            or end == len(patches)
            or (start // batch_size) % 20 == 0
        ):
            print(
                f"Processed "
                f"{end:,} / {len(patches):,} "
                f"patches"
            )

    predictions = np.concatenate(
        predictions
    )

    raw_log_variances = np.concatenate(
        raw_log_variances
    )

    # ------------------------------------------------------------
    # Convert raw log-variance output to variance
    #
    # This follows the same transformation used in
    # StableHeteroscedasticLoss:
    #
    # log_var = 5 * tanh(raw_log_var)
    # variance = exp(log_var)
    # ------------------------------------------------------------

    log_variance = (
        5.0 * np.tanh(raw_log_variances)
    )

    variance = np.exp(
        log_variance
    ).astype(np.float32)

    # ------------------------------------------------------------
    # Put predictions back into raster coordinates
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Put predictions back into raster coordinates
    # ------------------------------------------------------------

    if len(valid_positions) != len(predictions):
        raise ValueError(
            "Number of valid positions does not match "
            "number of predictions: "
            f"{len(valid_positions)} vs "
            f"{len(predictions)}"
        )

    prediction_map[
        valid_positions[:, 0],
        valid_positions[:, 1]
    ] = predictions

    variance_map[
        valid_positions[:, 0],
        valid_positions[:, 1]
    ] = variance

    return (
        prediction_map,
        variance_map,
        valid_mask
    )


# ================================================================
# Save GeoTIFF
# ================================================================

def save_geotiff(
    output_path,
    data,
    reference_dataset,
    nodata=-9999.0
):
    """
    Save a single-band GeoTIFF using the spatial metadata
    from the reference raster.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    profile = reference_dataset.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="deflate"
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            data.astype(np.float32),
            1
        )

    print(
        f"GeoTIFF saved to: "
        f"{output_path}"
    )


# ================================================================
# Main
# ================================================================

def main():

    print("=" * 60)
    print("SDB-PRO Full-Raster Prediction")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------

    config = load_config(
        PROJECT_ROOT
        / "configs"
        / "config.yaml"
    )

    raster_path = (
        PROJECT_ROOT
        / config["data"]["sentinel2"]["path"]
    )

    checkpoint_path = (
        PROJECT_ROOT
        / config["output"]["checkpoint_dir"]
        / "best_model.pth"
    )

    output_dir = (
        PROJECT_ROOT
        / config["output"]["prediction_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    bands = config["data"]["sentinel2"]["bands"]

    patch_size = config["model"]["patch_size"]

    batch_size = config["training"]["batch_size"]

    device = config["training"]["device"]

    # ------------------------------------------------------------
    # 2. Read Sentinel-2 raster
    # ------------------------------------------------------------

    print(
        f"\nInput raster: "
        f"{raster_path}"
    )

    print(
        f"Input bands: "
        f"{bands}"
    )

    with rasterio.open(raster_path) as src:

        image = src.read(
            bands
        ).astype(
            np.float32
        )

        print(
            f"Raster size: "
            f"{src.width} x {src.height}"
        )

        print(
            f"CRS: "
            f"{src.crs}"
        )

        print(
            f"Input shape: "
            f"{image.shape}"
        )

        # --------------------------------------------------------
        # 3. Create model
        # --------------------------------------------------------

        model = SDBCNN(
            in_channels=config["model"]["in_channels"],
            patch_size=patch_size
        )

        # --------------------------------------------------------
        # 4. Load trained model
        # --------------------------------------------------------

        predictor = SDBPredictor(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device
        )

        # --------------------------------------------------------
        # 5. Full-raster prediction
        # --------------------------------------------------------

        prediction_map, variance_map, valid_mask = (
            predict_raster(
                predictor=predictor,
                image=image,
                patch_size=patch_size,
                batch_size=batch_size
            )
        )

        # --------------------------------------------------------
        # 6. Save prediction GeoTIFF
        # --------------------------------------------------------

        prediction_path = (
            output_dir
            / "guam_sdb_prediction.tif"
        )

        save_geotiff(
            output_path=prediction_path,
            data=prediction_map,
            reference_dataset=src
        )

        # --------------------------------------------------------
        # 7. Save variance GeoTIFF
        # --------------------------------------------------------

        variance_path = (
            output_dir
            / "guam_sdb_variance.tif"
        )

        save_geotiff(
            output_path=variance_path,
            data=variance_map,
            reference_dataset=src
        )

    # ------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------

    valid_predictions = prediction_map[
        valid_mask
    ]

    valid_variance = variance_map[
        valid_mask
    ]

    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)

    print(
        f"Valid prediction pixels: "
        f"{len(valid_predictions):,}"
    )

    print(
        f"Prediction range: "
        f"{valid_predictions.min():.3f} ~ "
        f"{valid_predictions.max():.3f}"
    )

    print(
        f"Mean prediction: "
        f"{valid_predictions.mean():.3f}"
    )

    print(
        f"Variance range: "
        f"{valid_variance.min():.3f} ~ "
        f"{valid_variance.max():.3f}"
    )

    print(
        f"Prediction output: "
        f"{prediction_path}"
    )

    print(
        f"Variance output: "
        f"{variance_path}"
    )

    print("\nPrediction completed.")


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()