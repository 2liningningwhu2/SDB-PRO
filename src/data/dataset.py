from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SDBDataset(Dataset):
    """
    PyTorch Dataset for Satellite-Derived Bathymetry.

    从 .npz 文件读取：
        X           [N, C, H, W]
        y           [N]
        coordinates [N, 2]

    返回：
        X: Tensor [C, H, W]
        y: Tensor []
    """

    def __init__(self, npz_path):
        self.npz_path = Path(npz_path)

        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"Sample file not found: {self.npz_path}"
            )

        data = np.load(self.npz_path)

        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.coordinates = data["coordinates"].astype(
            np.float64
        )

        # --------------------------------------------------
        # 数据一致性检查
        # --------------------------------------------------

        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y have different lengths: "
                f"{len(self.X)} vs {len(self.y)}"
            )

        if len(self.X) != len(self.coordinates):
            raise ValueError(
                f"X and coordinates have different lengths: "
                f"{len(self.X)} vs "
                f"{len(self.coordinates)}"
            )

        if self.X.ndim != 4:
            raise ValueError(
                f"X must have 4 dimensions "
                f"[N, C, H, W], got {self.X.shape}"
            )

        print(
            f"Loaded dataset: {self.npz_path}"
        )

        print(
            f"X shape: {self.X.shape}"
        )

        print(
            f"y shape: {self.y.shape}"
        )

        print(
            f"coordinates shape: "
            f"{self.coordinates.shape}"
        )

    def __len__(self):
        """
        返回样本数量。
        """
        return len(self.X)

    def __getitem__(self, index):
        """
        获取一个训练样本。

        Returns
        -------
        x : torch.Tensor
            shape = [C, H, W]

        y : torch.Tensor
            scalar tensor
        """

        x = torch.from_numpy(
            self.X[index]
        )

        y = torch.tensor(
            self.y[index],
            dtype=torch.float32
        )

        return x, y

    def get_coordinates(self, index):
        """
        获取指定样本的经纬度。

        Returns
        -------
        np.ndarray
            [lon, lat]
        """
        return self.coordinates[index]
if __name__ == "__main__":

    dataset = SDBDataset(
        "outputs/samples/guam_samples.npz"
    )

    print("\n========== Dataset Test ==========")

    print(
        f"Dataset length: {len(dataset)}"
    )

    x, y = dataset[0]

    print(
        f"Single sample X shape: {x.shape}"
    )

    print(
        f"Single sample X dtype: {x.dtype}"
    )

    print(
        f"Single sample y shape: {y.shape}"
    )

    print(
        f"Single sample y: {y.item():.6f}"
    )

    print(
        f"Single sample coordinates: "
        f"{dataset.get_coordinates(0)}"
    )

    # --------------------------------------------------
    # DataLoader Test
    # --------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    X_batch, y_batch = next(iter(loader))

    print("\n========== DataLoader Test ==========")

    print(
        f"Batch X shape: {X_batch.shape}"
    )

    print(
        f"Batch X dtype: {X_batch.dtype}"
    )

    print(
        f"Batch y shape: {y_batch.shape}"
    )

    print(
        f"Batch y dtype: {y_batch.dtype}"
    )