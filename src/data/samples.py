from pathlib import Path

import numpy as np
import pandas as pd

from src.data.raster import RasterReader


class SampleBuilder:
    """
    构建卫星反演水深（SDB）训练样本。

    数据来源
    ----------
    Sentinel-2 GeoTIFF:
        用于提供多光谱影像 patch

    ICESat-2 CSV:
        用于提供样本位置和水深标签

    输出
    ----------
    X:
        shape = [N, C, H, W]

    y:
        shape = [N]

    coordinates:
        shape = [N, 2]
        每一行为 [lon, lat]
    """

    def __init__(
        self,
        raster_path,
        raster_bands,
        csv_path,
        longitude_column="lon_ph",
        latitude_column="lat_ph",
        depth_column="h_ph_refco",
        patch_size=5,
        remove_boundary_samples=True,
        remove_nan_samples=True,
    ):
        self.raster_path = Path(raster_path)
        self.csv_path = Path(csv_path)

        self.raster_bands = raster_bands

        self.longitude_column = longitude_column
        self.latitude_column = latitude_column
        self.depth_column = depth_column

        self.patch_size = patch_size

        self.remove_boundary_samples = remove_boundary_samples
        self.remove_nan_samples = remove_nan_samples

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"ICESat-2 CSV not found: {self.csv_path}"
            )

        if patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be an odd number, "
                "e.g. 3, 5, 7."
            )

    def load_icesat2(self):
        """
        读取 ICESat-2 样本 CSV。
        """

        df = pd.read_csv(self.csv_path)

        required_columns = [
            self.longitude_column,
            self.latitude_column,
            self.depth_column,
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # 只保留构建 SDB 样本所需要的字段
        df = df[required_columns].copy()

        # 删除经纬度或水深为空的记录
        df = df.dropna(
            subset=required_columns
        )

        # 删除无穷值
        df = df[
            np.isfinite(df[self.longitude_column])
            & np.isfinite(df[self.latitude_column])
            & np.isfinite(df[self.depth_column])
        ]

        df = df.reset_index(drop=True)

        print(
            f"Loaded ICESat-2 samples: {len(df)}"
        )

        return df

    def build(self):
        """
        构建完整 SDB 样本。

        Returns
        -------
        X : np.ndarray
            shape = [N, C, patch_size, patch_size]

        y : np.ndarray
            shape = [N]

        coordinates : np.ndarray
            shape = [N, 2]
            [lon, lat]
        """

        df = self.load_icesat2()

        X = []
        y = []
        coordinates = []

        removed_boundary = 0
        removed_nan = 0

        with RasterReader(
            self.raster_path,
            bands=self.raster_bands
        ) as raster:

            print(
                f"Raster size: "
                f"{raster.width} x {raster.height}"
            )

            print(
                f"Raster bands: "
                f"{self.raster_bands}"
            )

            for _, sample in df.iterrows():

                lon = float(
                    sample[self.longitude_column]
                )

                lat = float(
                    sample[self.latitude_column]
                )

                depth = float(
                    sample[self.depth_column]
                )

                # --------------------------------------------------
                # 1. 经纬度 → 栅格像素坐标
                # --------------------------------------------------
                row, col = raster.world_to_pixel(
                    lon,
                    lat
                )

                # --------------------------------------------------
                # 2. 判断中心点是否位于影像范围内
                # --------------------------------------------------
                if not raster.is_inside(row, col):
                    removed_boundary += 1
                    continue

                # --------------------------------------------------
                # 3. 提取 patch
                # --------------------------------------------------
                patch = raster.extract_patch(
                    row,
                    col,
                    patch_size=self.patch_size
                )

                # patch=None 表示靠近影像边界
                if patch is None:
                    removed_boundary += 1
                    continue

                # --------------------------------------------------
                # 4. 检查 NaN / Inf
                # --------------------------------------------------
                if self.remove_nan_samples:

                    if not np.all(np.isfinite(patch)):
                        removed_nan += 1
                        continue

                # --------------------------------------------------
                # 5. 保存样本
                # --------------------------------------------------
                X.append(patch.astype(np.float32))

                y.append(
                    np.float32(depth)
                )

                coordinates.append(
                    [
                        np.float64(lon),
                        np.float64(lat)
                    ]
                )

        if len(X) == 0:
            raise RuntimeError(
                "No valid samples were generated. "
                "Please check the raster, coordinates, "
                "and patch size."
            )

        X = np.stack(X, axis=0)
        y = np.asarray(y, dtype=np.float32)
        coordinates = np.asarray(
            coordinates,
            dtype=np.float64
        )

        print("\n========== Sample Summary ==========")

        print(
            f"Original ICESat-2 samples: "
            f"{len(df)}"
        )

        print(
            f"Removed boundary samples: "
            f"{removed_boundary}"
        )

        print(
            f"Removed NaN samples: "
            f"{removed_nan}"
        )

        print(
            f"Valid samples: "
            f"{len(X)}"
        )

        print(
            f"X shape: {X.shape}"
        )

        print(
            f"y shape: {y.shape}"
        )

        print(
            f"Depth range: "
            f"{y.min():.3f} ~ {y.max():.3f}"
        )

        return X, y, coordinates

    @staticmethod
    def save(
        output_path,
        X,
        y,
        coordinates
    ):
        """
        保存样本数据。

        使用 npz 保存：
            X
            y
            coordinates
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            coordinates=coordinates
        )

        print(
            f"\nSamples saved to: "
            f"{output_path}"
        )

        print(
            f"File size: "
            f"{output_path.stat().st_size / 1024 / 1024:.2f} MB"
        )


if __name__ == "__main__":

    builder = SampleBuilder(
        raster_path="data/guam_S2_8band.tif",
        raster_bands=[2, 3, 4, 8],
        csv_path="data/keypoints_10.csv",
        longitude_column="lon_ph",
        latitude_column="lat_ph",
        depth_column="h_ph_refco",
        patch_size=5,
    )

    X, y, coordinates = builder.build()

    builder.save(
        output_path="outputs/samples/guam_samples.npz",
        X=X,
        y=y,
        coordinates=coordinates,
    )