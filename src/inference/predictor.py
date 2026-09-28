from pathlib import Path

import numpy as np
import torch


class SDBPredictor:
    """
    CNN predictor for Satellite-Derived Bathymetry.

    Responsibilities
    ----------------
    1. Load a trained model checkpoint.
    2. Run batch inference.
    3. Return predicted depth and raw uncertainty-related output.

    Model output
    ------------
    output[:, 0]:
        Predicted depth.

    output[:, 1]:
        Raw log-variance-related output from the network.

        Note:
        The loss function applies the transformation

            log_var = scale * tanh(raw_log_var)

        during training. Therefore, this class returns the raw
        network output and does not call it "log variance".
    """

    def __init__(self, model, checkpoint_path, device="auto"):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model architecture.

        checkpoint_path : str or Path
            Path to the trained model checkpoint.

        device : str
            "auto", "cpu", or "cuda".
        """
        self.model = model

        # ---------------------------------------------------------
        # Select inference device
        # ---------------------------------------------------------
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ---------------------------------------------------------
        # Check checkpoint
        # ---------------------------------------------------------
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

        # ---------------------------------------------------------
        # Load checkpoint
        # ---------------------------------------------------------
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'model_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.checkpoint = checkpoint

        print(f"Loaded checkpoint: {checkpoint_path}")

        if "epoch" in checkpoint:
            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_loss" in checkpoint:
            print(
                f"Checkpoint val_loss: "
                f"{checkpoint['val_loss']:.6f}"
            )

        print(
            f"Inference device: {self.device}"
        )

    @torch.no_grad()
    def predict(self, data_loader):
        """
        Run batch inference.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader containing SDB input patches.

        Returns
        -------
        predictions : np.ndarray
            Predicted depth values, shape [N].

        raw_log_variance : np.ndarray
            Raw second output of the network, shape [N].
        """

        all_predictions = []
        all_raw_log_variance = []

        # ---------------------------------------------------------
        # Batch inference
        # ---------------------------------------------------------
        for X, _ in data_loader:

            X = X.to(self.device)

            output = self.model(X)

            # First output: predicted depth
            depth = output[:, 0]

            # Second output: raw uncertainty-related output
            raw_log_var = output[:, 1]

            all_predictions.append(
                depth.cpu().numpy()
            )

            all_raw_log_variance.append(
                raw_log_var.cpu().numpy()
            )

        # ---------------------------------------------------------
        # Merge all batches
        # ---------------------------------------------------------
        if len(all_predictions) == 0:
            raise ValueError(
                "DataLoader produced no batches."
            )

        predictions = np.concatenate(
            all_predictions
        )

        raw_log_variance = np.concatenate(
            all_raw_log_variance
        )

        return predictions, raw_log_variance


# =================================================================
# Standalone test
# =================================================================

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    from src.data.dataset import SDBDataset
    from src.models.cnn import SDBCNN

    print("========== Predictor Test ==========")

    # -------------------------------------------------------------
    # 1. Load dataset
    # -------------------------------------------------------------
    dataset = SDBDataset(
        "outputs/samples/guam_samples.npz"
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    # -------------------------------------------------------------
    # 2. Create model
    # -------------------------------------------------------------
    model = SDBCNN(
        in_channels=4,
        patch_size=5
    )

    # -------------------------------------------------------------
    # 3. Create predictor and load checkpoint
    # -------------------------------------------------------------
    predictor = SDBPredictor(
        model=model,
        checkpoint_path="checkpoints/best_model.pth",
        device="auto"
    )

    # -------------------------------------------------------------
    # 4. Run inference
    # -------------------------------------------------------------
    predictions, raw_log_variance = predictor.predict(
        loader
    )

    # -------------------------------------------------------------
    # 5. Print results
    # -------------------------------------------------------------
    print(
        f"\nPrediction shape: "
        f"{predictions.shape}"
    )

    print(
        f"Raw log variance shape: "
        f"{raw_log_variance.shape}"
    )

    print(
        f"Prediction range: "
        f"{predictions.min():.3f} ~ "
        f"{predictions.max():.3f}"
    )

    print(
        f"Raw log variance range: "
        f"{raw_log_variance.min():.3f} ~ "
        f"{raw_log_variance.max():.3f}"
    )

    print(
        f"\nNumber of predictions: "
        f"{len(predictions)}"
    )

    print("\nPredictor test passed.")