from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy


class RasterReader:
    """
    Sentinel-2 / GeoTIFF 栅格数据读取器。

    主要负责：
    1. 打开 GeoTIFF
    2. 读取指定波段
    3. 获取 CRS、transform、尺寸等空间信息
    4. 经纬度与像素坐标之间转换
    5. 提取局部 patch
    """

    def __init__(self, path, bands=None):
        """
        Parameters
        ----------
        path : str or Path
            GeoTIFF 文件路径

        bands : list[int], optional
            需要读取的波段编号，例如 [1, 2, 3, 4]
        """
        self.path = Path(path)
        self.bands = bands

        if not self.path.exists():
            raise FileNotFoundError(
                f"Raster file not found: {self.path}"
            )

        self.dataset = rasterio.open(self.path)

        self.width = self.dataset.width
        self.height = self.dataset.height
        self.count = self.dataset.count
        self.crs = self.dataset.crs
        self.transform = self.dataset.transform
        self.dtype = self.dataset.dtypes[0]

    def read(self):
        """
        读取指定波段。

        Returns
        -------
        np.ndarray
            shape = [C, H, W]
        """
        if self.bands is None:
            data = self.dataset.read()
        else:
            data = self.dataset.read(self.bands)

        return data

    def world_to_pixel(self, lon, lat):
        """
        将地理坐标转换为栅格行列号。

        Parameters
        ----------
        lon : float
            经度

        lat : float
            纬度

        Returns
        -------
        row : int
            行号

        col : int
            列号
        """
        row, col = rowcol(
            self.transform,
            lon,
            lat
        )

        return int(row), int(col)

    def pixel_to_world(self, row, col):
        """
        将栅格行列号转换为地理坐标。

        Parameters
        ----------
        row : int
            行号

        col : int
            列号

        Returns
        -------
        lon : float
            经度

        lat : float
            纬度
        """
        lon, lat = xy(
            self.transform,
            row,
            col,
            offset="center"
        )

        return lon, lat

    def is_inside(self, row, col):
        """
        判断像素坐标是否位于影像范围内。
        """
        return (
            0 <= row < self.height
            and 0 <= col < self.width
        )

    def extract_patch(self, row, col, patch_size=5):
        """
        提取以(row, col)为中心的局部 patch。

        Parameters
        ----------
        row : int
            中心像素行号

        col : int
            中心像素列号

        patch_size : int
            Patch尺寸，例如5表示5×5

        Returns
        -------
        np.ndarray or None
            shape = [C, patch_size, patch_size]

            如果patch超出影像范围，则返回None。
        """
        if patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be an odd number, "
                "e.g. 3, 5, 7."
            )

        half = patch_size // 2

        row_start = row - half
        row_end = row + half + 1

        col_start = col - half
        col_end = col + half + 1

        if (
            row_start < 0
            or row_end > self.height
            or col_start < 0
            or col_end > self.width
        ):
            return None

        data = self.read_window(
            row_start,
            row_end,
            col_start,
            col_end
        )

        return data

    def read_window(
        self,
        row_start,
        row_end,
        col_start,
        col_end
    ):
        """
        读取指定像素窗口。

        Returns
        -------
        np.ndarray
            shape = [C, H, W]
        """
        window = rasterio.windows.Window(
            col_start,
            row_start,
            col_end - col_start,
            row_end - row_start
        )

        if self.bands is None:
            data = self.dataset.read(window=window)
        else:
            data = self.dataset.read(
                self.bands,
                window=window
            )

        return data

    def close(self):
        """
        关闭 GeoTIFF 数据集。
        """
        if self.dataset:
            self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()