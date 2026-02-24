"""
download_tile.py

Downloads a real Sentinel-2 satellite image patch from Google Earth Engine
as an RGB PNG (for SAM segmentation) and triggers a GeoTIFF export to Drive.

Usage:
    python download_tile.py                              # Bengaluru (default)
    python download_tile.py --city Mumbai
    python download_tile.py --lon 72.90 --lat 19.08 --city Mumbai
"""

import os
import ee
import logging
import argparse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── City registry (lon_center, lat_center, half-size degrees) ─────────────────
CITY_COORDS = {
    "Bengaluru":  (77.620, 12.965, 0.03),
    "Mumbai":     (72.880, 19.090, 0.03),
    "Delhi":      (77.230, 28.613, 0.03),
    "Chennai":    (80.270, 13.085, 0.03),
    "Hyderabad":  (78.470, 17.385, 0.03),
    "Pune":       (73.856, 18.520, 0.03),
}

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--city", default="Bengaluru", choices=list(CITY_COORDS.keys()))
parser.add_argument("--lon",  type=float, default=None, help="Centre longitude override")
parser.add_argument("--lat",  type=float, default=None, help="Centre latitude override")
parser.add_argument("--year", default="2024")
args = parser.parse_args()

city_name = args.city
lon, lat, half = CITY_COORDS[city_name]
if args.lon: lon = args.lon
if args.lat: lat = args.lat

# ── Init GEE ─────────────────────────────────────────────────────────────────
load_dotenv()
ee.Initialize(project=os.environ.get("EE_PROJECT_ID", "geosight-project"))

PATCH   = ee.Geometry.Rectangle([lon-half, lat-half, lon+half, lat+half])
slug    = city_name.lower()
OUT_DIR = Path(f"data/{slug}_patch")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG  = str(OUT_DIR / f"{slug}_rgb.png")

# ── Find best scene ─────────────────────────────────────────────────────────
logger.info(f"Finding clearest Sentinel-2 scene over {city_name} ({args.year})…")
img = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(PATCH)
    .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .sort("CLOUDY_PIXEL_PERCENTAGE")
    .first()
)
date = img.date().format("YYYY-MM-dd").getInfo()
logger.info(f"Best scene date: {date}")

# ── Download RGB thumbnail ───────────────────────────────────────────────────
logger.info("Downloading RGB thumbnail (2048×2048)…")
url = img.getThumbURL({
    "dimensions": "2048x2048",
    "region":     PATCH,
    "bands":      ["B4", "B3", "B2"],
    "min":        0, "max": 3000,
    "format":     "png", "gamma": 1.4
})
urllib.request.urlretrieve(url, OUT_PNG)
logger.info(f"✅ RGB patch saved: {OUT_PNG}")

# ── Start GeoTIFF export to Drive ────────────────────────────────────────────
logger.info("Triggering multi-band GeoTIFF export to Google Drive…")
task = ee.batch.Export.image.toDrive(
    image=img.select(["B2","B3","B4","B8","B11"]),
    description=f"{slug}_patch_{args.year}",
    fileNamePrefix=f"{slug}_bands",
    region=PATCH, scale=10, crs="EPSG:32643",
    fileFormat="GeoTIFF", maxPixels=1e9,
)
task.start()
logger.info(f"✅ GeoTIFF export started (GEE Task ID: {task.id})")
logger.info("   View at: https://code.earthengine.google.com/tasks")
logger.info(f"\n✅ {city_name} tile ready — run sam2_segmentation.py --city {city_name} next!")

# Print the output path so app.py can capture it
print(f"TILE_PATH={OUT_PNG}")
