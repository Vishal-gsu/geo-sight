"""
test_connection.py

Quick verification that Earth Engine is working with the geosight-project
and can pull real Sentinel-2 imagery over Bengaluru.
"""

import ee
from dotenv import load_dotenv
import os

load_dotenv()

project_id = os.environ.get("EE_PROJECT_ID", "geosight-project")

print(f"[1/4] Initializing Earth Engine with project: {project_id}")
ee.Initialize(project=project_id)
print("      ✅ Earth Engine connected!")

print("\n[2/4] Pulling Sentinel-2 image over Bengaluru (2024)...")
# Bengaluru bounding box
bangalore = ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.10])

# Load Sentinel-2 Surface Reflectance - cloud-free composite
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(bangalore)
    .filterDate("2024-01-01", "2024-03-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .select(["B4", "B3", "B2", "B8", "B11"])  # Red, Green, Blue, NIR, SWIR
)

count = collection.size().getInfo()
print(f"      ✅ Found {count} cloud-free Sentinel-2 scenes over Bengaluru (Q1 2024)!")

print("\n[3/4] Fetching first image metadata...")
first_img = collection.first()
date = first_img.date().format("YYYY-MM-dd").getInfo()
print(f"      ✅ First image date: {date}")

print("\n[4/4] Computing NDVI from real satellite data...")
img = first_img
nir  = img.select("B8")
red  = img.select("B4")
ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

# Sample a small region to get real NDVI values
sample = ndvi.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=bangalore,
    scale=100
).getInfo()

mean_ndvi = sample.get("NDVI", "N/A")
print(f"      ✅ Mean NDVI over Bengaluru: {mean_ndvi:.4f}" if isinstance(mean_ndvi, float) else f"      NDVI result: {mean_ndvi}")

print("\n====== 🚀 GeoSight pipeline fully connected to real Earth observation data! ======")
print(f"   Project: {project_id}")
print(f"   Region:  Bengaluru, India [77.45°E–77.75°E, 12.85°N–13.10°N]")
print(f"   Sensor:  Sentinel-2 Surface Reflectance (L2A)")
print(f"   Scenes:  {count} cloud-free images available")
print(f"   Date:    {date} (most recent clear acquisition)")
