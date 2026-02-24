# 🛰️ GeoSight — Multi-Layer Satellite Image Analysis
### Google Earth Engine · SAM · Sentinel-2 · Bengaluru Land Cover Change Detection

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)
![GEE](https://img.shields.io/badge/Google_Earth_Engine-Free_Tier-4285F4?style=flat-square&logo=google)
![SAM2](https://img.shields.io/badge/SAM_2-Meta_AI-0064E0?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An end-to-end geospatial image analysis pipeline combining **Google Earth Engine (Sentinel-2 L2A)** with **Meta's SAM (Segment Anything Model) via samgeo** for zero-shot land cover segmentation and multi-temporal change detection over Bengaluru, India (2019 → 2024).

> **Note on AlphaEarth Foundations:** AlphaEarth is a Google DeepMind Foundation Model for geospatial embeddings. Its public GEE dataset collection is not yet available for academic accounts (as of early 2025). This project is architected to integrate AlphaEarth the moment it becomes accessible — `src/embeddings.py` already contains the planned integration pattern.

---

## 📊 Real Results — Bengaluru 2019 → 2024

| Metric | 2019 | 2024 | Change |
|:---|:---:|:---:|:---:|
| Mean NDVI | 0.2598 | 0.2507 | **–0.0091** ⚠️ |
| Mean NDBI | 0.0513 | 0.0514 | **+0.0001** 🏙️ |

> **Finding:** Consistent aggregate vegetation decline across the Bengaluru metro area, indicative of continued IT corridor expansion into peripheral greenbelt zones.

---

## 🗂️ Project Structure

```
geosight/
├── src/
│   ├── preprocess.py      # Rasterio loading · cloud masking · UTM reprojection
│   ├── spectral.py        # NDVI · NDWI · NDBI band math
│   ├── segmentation.py    # SAM 2 auto-segmentation + spectral class labelling
│   ├── change_detect.py   # Multi-temporal difference analysis
│   ├── embeddings.py      # Google Earth Engine / AlphaEarth API
│   └── report.py          # Dark-mode matplotlib figures · Folium maps · area stats
├── notebooks/
│   └── geosight_demo.ipynb   # Interactive end-to-end walkthrough
├── data/                  # Downloaded satellite patches (gitignored)
├── results/               # Generated maps, PNGs, HTML, JSON stats
├── main.py                # Synthetic data demonstration
├── bangalore_change_detection.py  # Full real-data pipeline
├── download_tile.py       # GEE satellite tile downloader
├── sam2_segmentation.py   # SAM 2 zero-shot segmentation runner
├── test_connection.py     # Earth Engine connection test
├── run.ps1                # One-click Windows setup
├── .env.example           # Environment variable template
└── requirements.txt
```

---

## 🚀 One-Click Setup (Windows)

```powershell
# Clone & enter project
git clone https://github.com/YOUR_USERNAME/geosight
cd geosight

# One-click: creates venv, installs deps, creates .env
.\run.ps1

# Authenticate Earth Engine (first time only)
earthengine authenticate

# Edit .env → add your GEE project ID
notepad .env
```

### Manual (Linux/Mac)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env
earthengine authenticate
```

---

## 🔬 Pipeline Overview

```
Sentinel-2 L2A (13 bands, 10m)
         │
         ▼
 Preprocessing (preprocess.py)
 ├── SCL cloud masking
 ├── Percentile normalization
 └── UTM reprojection (EPSG:32643)
         │
         ▼
 Spectral Indices (spectral.py)
 ├── NDVI = (NIR–Red)/(NIR+Red)
 ├── NDWI = (Green–NIR)/(Green+NIR)
 └── NDBI = (SWIR–NIR)/(SWIR+NIR)
         │
         ├──────────────────────┐
         ▼                      ▼
 SAM 2 Segmentation       Change Detection
 (segmentation.py)        (change_detect.py)
 ├── Zero-shot masks       ├── ΔIndex T1→T2
 ├── Spectral classify     ├── Deforestation mask
 └── Morphological clean   └── Urban expansion mask
         │                      │
         └──────────┬───────────┘
                    ▼
             Report Generation (report.py)
             ├── Dark-mode 6-panel PNG
             ├── Interactive Folium HTML
             └── Area statistics JSON
```

---

## ⚡ Quick Run

```bash
# Test Earth Engine connection
python test_connection.py

# Run synthetic demo (no GEE needed)
python main.py

# Full real-data pipeline: Bengaluru 2019 → 2024
python bangalore_change_detection.py

# Download satellite tile (for SAM 2)
python download_tile.py

# Run SAM 2 zero-shot segmentation  
python sam2_segmentation.py

# Open interactive notebook
jupyter notebook notebooks/geosight_demo.ipynb
```

---

## 🔑 Environment Configuration

Copy `.env.example` → `.env` and set your project:
```env
EE_PROJECT_ID=your-gee-project-id
EE_SERVICE_ACCOUNT=          # Optional: for production deployment
EE_PRIVATE_KEY_PATH=         # Optional: for production deployment
```

---

## 📚 Tech Stack

| Category | Libraries |
|:---|:---|
| **Geospatial** | `earthengine-api`, `rasterio`, `geopandas`, `GDAL` |
| **Satellite Data** | Sentinel-2 L2A via Google Earth Engine (free, non-commercial) |
| **Segmentation** | Meta SAM `vit_b` via `segment-geospatial` |
| **Image Processing** | `numpy`, `opencv-python`, `scikit-image` |
| **Visualization** | `matplotlib`, `folium`, `leafmap` |
| **Configuration** | `python-dotenv` |
| **Testing** | `pytest`, `pytest-cov` |

---

## 🎯 Job Description Alignment (KaleidEO / SatSure)

| JD Requirement | Implementation |
|:---|:---|
| Image preprocessing & quality checks | `preprocess.py` — cloud masking, normalization, UTM reproject |
| Geospatial libraries (rasterio, GDAL) | Fully integrated in preprocessing pipeline |
| Spectral / frequency domain concepts | NDVI/NDWI/NDBI + 3 deep-dive learning documents |
| Deep Learning exposure (SAM, ViT) | `sam2_segmentation.py` — SAM `vit_b` zero-shot masks + spectral classification |
| Change detection & report generation | `bangalore_change_detection.py` + Folium HTML + JSON stats |
| Python scripting & automation | Modular `src/` package with logging, type hints, error handling |
| Code quality | `pytest` unit tests + `ruff` linting + `.gitignore` + `requirements-dev.txt` |
| AlphaEarth / Foundation models | `src/embeddings.py` — integration pattern ready, awaiting public dataset release |

---

*Built as a portfolio project targeting Earth Observation and Remote Sensing roles.  
CMR Institute of Technology | Google Earth Engine Non-Commercial Academic License*
