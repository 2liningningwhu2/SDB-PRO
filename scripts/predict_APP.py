import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import torch


# ============================================================
# Project / Application Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# PyInstaller 打包后：
#   sys.executable -> SDB-PRO.exe
#   application root -> exe 所在目录
#
# 源码运行时：
#   application root -> 项目根目录
if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = PROJECT_ROOT


# 让 Python 可以找到 src/
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn import SDBCNN
from src.inference.predictor import SDBPredictor


# ============================================================
# Path Utilities
# ============================================================

def resolve_path(path_text, base_dir):
    """
    将 XML 中的路径解析为绝对路径。

    规则：
    1. 绝对路径：直接使用
    2. 相对路径：相对于 APP_ROOT
    """
    path = Path(path_text.strip())

    if path.is_absolute():
        return path

    return (base_dir / path).resolve()


# ============================================================
# XML Configuration
# ============================================================

def read_xml_config(config_path):
    """
    读取 XML 配置，并将其中所有路径解析为绝对路径。
    """

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    tree = ET.parse(config_path)
    root = tree.getroot()

    if root.tag != "prediction":
        raise ValueError(
            "Invalid XML configuration: root tag must be <prediction>"
        )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_node = root.find("input")

    if input_node is None:
        raise ValueError("Missing <input> section in XML")

    raster_node = input_node.find("raster")
    checkpoint_node = input_node.find("checkpoint")

    if raster_node is None or raster_node.text is None:
        raise ValueError("Missing <raster> in XML")

    if checkpoint_node is None or checkpoint_node.text is None:
        raise ValueError("Missing <checkpoint> in XML")

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_node = root.find("output")

    if output_node is None:
        raise ValueError("Missing <output> section in XML")

    prediction_node = output_node.find("prediction")
    variance_node = output_node.find("variance")

    if prediction_node is None or prediction_node.text is None:
        raise ValueError("Missing <prediction> in XML")

    if variance_node is None or variance_node.text is None:
        raise ValueError("Missing <variance> in XML")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_node = root.find("model")

    if model_node is None:
        raise ValueError("Missing <model> section in XML")

    bands_node = model_node.find("bands")
    patch_size_node = model_node.find("patch_size")
    batch_size_node = model_node.find("batch_size")

    if bands_node is None or bands_node.text is None:
        raise ValueError("Missing <bands> in XML")

    if patch_size_node is None or patch_size_node.text is None:
        raise ValueError("Missing <patch_size> in XML")

    if batch_size_node is None or batch_size_node.text is None:
        raise ValueError("Missing <batch_size> in XML")

    # --------------------------------------------------------
    # Parse values
    # --------------------------------------------------------

    bands = [
        int(value.strip())
        for value in bands_node.text.split(",")
        if value.strip()
    ]

    patch_size = int(patch_size_node.text.strip())
    batch_size = int(batch_size_node.text.strip())

    if not bands:
        raise ValueError("At least one input band is required")

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(
            "patch_size must be a positive odd integer"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0"
        )

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    raster_path = resolve_path(
        raster_node.text,
        APP_ROOT
    )

    checkpoint_path = resolve_path(
        checkpoint_node.text,
        APP_ROOT
    )

    prediction_path = resolve_path(
        prediction_node.text,
        APP_ROOT
    )

    variance_path = resolve_path(
        variance_node.text,
        APP_ROOT
    )

    return {
        "raster": raster_path,
        "checkpoint": checkpoint_path,
        "prediction": prediction_path,
        "variance": variance_path,
        "bands": bands,
        "patch_size": patch_size,
        "batch_size": batch_size,
    }


# ============================================================
# Sliding Window
# ============================================================

def create_patches(raster_data, patch_size):
    """
    Create sliding-window patches from raster data.
    """

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(
            "patch_size must be a positive odd integer"
        )

    _, height, width = raster_data.shape
    half = patch_size // 2

    patches = []
    positions = []

    for row in range(half, height - half):
        for col in range(half, width - half):

            patch = raster_data[
                :,
                row - half:row + half + 1,
                col - half:col + half + 1
            ]

            # Skip patches containing NaN / Inf
            if not np.isfinite(patch).all():
                continue

            patches.append(patch)
            positions.append((row, col))

    if not patches:
        raise RuntimeError(
            "No valid patches were generated from the input raster."
        )

    return (
        np.asarray(patches, dtype=np.float32),
        np.asarray(positions, dtype=np.int32),
    )


# ============================================================
# Raster Prediction
# ============================================================

def predict_raster(
    input_path,
    checkpoint_path,
    output_path,
    variance_output_path,
    bands,
    patch_size,
    batch_size,
):

    input_path = Path(input_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    variance_output_path = Path(variance_output_path)

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input raster not found:\n{input_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("=" * 60)
    print("SDB-PRO Application Inference")
    print("=" * 60)

    print(f"\nApplication root: {APP_ROOT}")
    print(f"Input raster: {input_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prediction output: {output_path}")
    print(f"Variance output: {variance_output_path}")
    print(f"Input bands: {bands}")
    print(f"Patch size: {patch_size}")
    print(f"Batch size: {batch_size}")

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    print("\nReading input raster...")

    with rasterio.open(input_path) as src:

        raster_data = src.read(bands)

        profile = src.profile.copy()

        height = src.height
        width = src.width
        crs = src.crs

    print(f"Raster size: {width} x {height}")
    print(f"CRS: {crs}")
    print(f"Input shape: {raster_data.shape}")

    # --------------------------------------------------------
    # Create patches
    # --------------------------------------------------------

    print("\nCreating sliding-window patches...")

    patches, positions = create_patches(
        raster_data,
        patch_size
    )

    print(f"Valid patches: {len(patches):,}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = SDBCNN(
        in_channels=len(bands),
        patch_size=patch_size
    )

    predictor = SDBPredictor(
        model=model,
        checkpoint_path=checkpoint_path,
        device="auto"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning CNN inference...")

    predictions = []
    raw_uncertainty = []

    total = len(patches)

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        batch = torch.from_numpy(
            patches[start:end]
        ).to(predictor.device)

        with torch.no_grad():

            output = predictor.model(batch)

        pred = (
            output[:, 0]
            .cpu()
            .numpy()
        )

        raw_var = (
            output[:, 1]
            .cpu()
            .numpy()
        )

        predictions.append(pred)
        raw_uncertainty.append(raw_var)

        print(
            f"\rProcessed {end:,} / {total:,} patches",
            end=""
        )

    print()

    predictions = np.concatenate(
        predictions
    )

    raw_uncertainty = np.concatenate(
        raw_uncertainty
    )

    # --------------------------------------------------------
    # Convert uncertainty
    # --------------------------------------------------------

    log_var = 5.0 * np.tanh(
        raw_uncertainty
    )

    variance = np.exp(log_var)

    # --------------------------------------------------------
    # Create output maps
    # --------------------------------------------------------

    prediction_map = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    variance_map = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    rows = positions[:, 0]
    cols = positions[:, 1]

    if len(predictions) != len(positions):
        raise RuntimeError(
            "Prediction count does not match "
            "valid patch count."
        )

    prediction_map[
        rows,
        cols
    ] = predictions

    variance_map[
        rows,
        cols
    ] = variance

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    variance_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # GeoTIFF profile
    # --------------------------------------------------------

    output_profile = profile.copy()

    output_profile.update(
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate"
    )

    # --------------------------------------------------------
    # Write prediction
    # --------------------------------------------------------

    with rasterio.open(
        output_path,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            prediction_map,
            1
        )

    # --------------------------------------------------------
    # Write variance
    # --------------------------------------------------------

    with rasterio.open(
        variance_output_path,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            variance_map,
            1
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    valid_prediction = prediction_map[
        np.isfinite(prediction_map)
    ]

    valid_variance = variance_map[
        np.isfinite(variance_map)
    ]

    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)

    print(
        f"Valid prediction pixels: "
        f"{len(valid_prediction):,}"
    )

    print(
        f"Prediction range: "
        f"{valid_prediction.min():.3f} "
        f"~ {valid_prediction.max():.3f}"
    )

    print(
        f"Mean prediction: "
        f"{valid_prediction.mean():.3f}"
    )

    print(
        f"Variance range: "
        f"{valid_variance.min():.3f} "
        f"~ {valid_variance.max():.3f}"
    )

    print(
        f"Prediction output: "
        f"{output_path}"
    )

    print(
        f"Variance output: "
        f"{variance_output_path}"
    )

    print("\nPrediction completed.")


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python scripts\\predict_app.py "
            "configs\\predict.xml"
        )

        sys.exit(1)

    config_path = Path(
        sys.argv[1]
    ).resolve()

    print(
        f"Configuration: {config_path}"
    )

    config = read_xml_config(
        config_path
    )

    predict_raster(
        input_path=config["raster"],
        checkpoint_path=config["checkpoint"],
        output_path=config["prediction"],
        variance_output_path=config["variance"],
        bands=config["bands"],
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
    )


if __name__ == "__main__":
    main()