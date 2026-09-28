# SDB-PRO

SDB-PRO is a practical deep-learning application for shallow-water
Satellite-Derived Bathymetry (SDB).

The project uses Sentinel-2 multispectral imagery as input and
ICESat-2 bathymetric measurements as supervision. A lightweight
2D convolutional neural network (CNN) is trained to learn the
relationship between multispectral image patches and water depth,
and the trained model can then be used to generate full-raster
bathymetry products.

The project is designed as an end-to-end remote-sensing AI
application, covering data preparation, model training,
evaluation, raster inference, configuration management, and
Windows executable deployment.

---

## 1. Features

- Sentinel-2 GeoTIFF data reading with Rasterio
- ICESat-2 bathymetric sample processing
- 5 × 5 multispectral image patch construction
- PyTorch CNN-based water-depth inversion
- Heteroscedastic regression loss
- Training and validation
- RMSE, MAE, R² and bias evaluation
- Full-raster sliding-window inference
- GeoTIFF bathymetry output
- XML-based inference configuration
- Windows executable deployment with PyInstaller
- Portable `onedir` application deployment

---

## 2. Workflow

The complete workflow is:

```text
Sentinel-2 GeoTIFF
        │
        │
        ├──────────────┐
        │              │
        ▼              ▼
Multispectral      ICESat-2 CSV
Raster Data        Bathymetry
        │              │
        └──────┬───────┘
               ▼
        Sample Construction
               │
               ▼
       5 × 5 Image Patches
               │
               ▼
          PyTorch CNN
               │
               ▼
       Model Training
               │
               ▼
        Model Evaluation
               │
               ▼
       Trained Checkpoint
               │
               ▼
     Full-Raster Inference
               │
               ▼
       Bathymetry GeoTIFF