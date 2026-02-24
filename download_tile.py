"""
download_tile.py

Downloads a real Sentinel-2 satellite image patch over central Bengaluru
from Google Earth Engine as a GeoTIFF, ready for SAM 2 segmentation.

Run this BEFORE sam2_segmentation.py.
"""

import os
import ee
import logging
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
ee.Initialize(project=os.environ.get("EE_PROJECT_ID", "geosight-project"))

# ── Patch over central Bengaluru (Koramangala / Indiranagar area) ───────────
PATCH = ee.Geometry.Rectangle([77.59, 12.94, 77.65, 12.99])  # ~6x5 km
OUT_DIR = Path("data/bangalore_patch")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG  = str(OUT_DIR / "bangalore_rgb.png")
OUT_TIFF = str(OUT_DIR / "bangalore_bands.tif")   # for NDVI/NDWI later


def get_best_image(year: str) -> ee.Image:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(PATCH)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )


logger.info("Finding clearest Sentinel-2 scene over central Bengaluru (2024)…")
img = get_best_image("2024")
date = img.date().format("YYYY-MM-dd").getInfo()
logger.info(f"Best scene date: {date}")

# ── Download RGB thumbnail as PNG (for SAM 2 input) ─────────────────────────
logger.info("Downloading RGB thumbnail…")
thumb_params = {
    "dimensions":  "2048x2048",
    "region":      PATCH,
    "bands":       ["B4", "B3", "B2"],       # Red, Green, Blue
    "min":         0,
    "max":         3000,
    "format":      "png",
    "gamma":       1.4
}
url = img.getThumbURL(thumb_params)
urllib.request.urlretrieve(url, OUT_PNG)
logger.info(f"✅ RGB patch saved: {OUT_PNG}")

# ── Download multi-band for spectral indices ──────────────────────────────────
logger.info("Downloading multi-band GeoTIFF for spectral analysis…")
task = ee.batch.Export.image.toDrive(
    image=img.select(["B2", "B3", "B4", "B8", "B11"]),
    description="bangalore_patch_2024",
    fileNamePrefix="bangalore_bands",
    region=PATCH,
    scale=10,
    crs="EPSG:32643",
    fileFormat="GeoTIFF",
    maxPixels=1e9,
)
task.start()
logger.info(f"✅ GeoTIFF export task started in GEE (ID: {task.id})")
logger.info("   Check status at: https://code.earthengine.google.com/tasks")
logger.info("   GeoTIFF will be saved to your Google Drive when complete.")
logger.info("\n✅ RGB PNG is available immediately for SAM 2 — run sam2_segmentation.py next!")
