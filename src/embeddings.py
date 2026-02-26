"""
embeddings.py

Module to interact with the Google Earth Engine Python API to access
satellite imagery and run server-side analysis over Bengaluru.

NOTE on AlphaEarth Foundations:
    AlphaEarth Foundation Model embeddings are a Google DeepMind product
    integrated into Earth Engine. As of early 2026, the public GEE dataset
    collection ID is not yet finalized/available for academic accounts.
    This module is built ready to integrate AlphaEarth the moment the
    collection becomes available. Until then, we use Sentinel-2 L2A bands
    directly for all spectral analysis.

    Reference: https://earthengine.google.com/alpha_earth (availability TBD)
"""

import ee
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def authenticate_and_initialize():
    """
    Authenticates and initializes the Earth Engine API session.

    - If EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_PATH are set in .env → uses service account (production).
    - Otherwise → uses local credentials saved by 'earthengine authenticate' (development).
    """
    try:
        project_id      = os.environ.get("EE_PROJECT_ID")
        service_account = os.environ.get("EE_SERVICE_ACCOUNT")
        key_path        = os.environ.get("EE_PRIVATE_KEY_PATH")

        if service_account and key_path:
            logger.info("Initializing Earth Engine with Service Account credentials.")
            credentials = ee.ServiceAccountCredentials(service_account, key_path)
            ee.Initialize(credentials, project=project_id)
        else:
            logger.info("Initializing Earth Engine with local credentials (earthengine authenticate).")
            ee.Initialize(project=project_id)

        logger.info(f"✅ Earth Engine initialized — project: {project_id}")

    except Exception as e:
        logger.error(
            f"Earth Engine initialization failed. "
            f"Run 'earthengine authenticate' for first-time setup. Error: {e}"
        )
        raise


def get_sentinel2_composite(
    roi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: float = 15.0,
    bands: list = None
) -> ee.Image:
    """
    Returns a median cloud-free Sentinel-2 L2A composite for the given ROI and date range.

    Args:
        roi         : Earth Engine Geometry (e.g. ee.Geometry.Rectangle)
        start_date  : ISO date string e.g. "2024-01-01"
        end_date    : ISO date string e.g. "2024-12-31"
        cloud_pct   : Maximum CLOUDY_PIXEL_PERCENTAGE (0–100)
        bands       : List of band names to select (default: RGB + NIR + SWIR)

    Returns:
        Median composite ee.Image
    """
    if bands is None:
        bands = ["B2", "B3", "B4", "B8", "B11"]  # Blue, Green, Red, NIR, SWIR

    logger.info(f"Building S2 composite: {start_date} → {end_date}, cloud_pct ≤ {cloud_pct}%")
    composite = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select(bands)
        .median()
    )
    return composite


def fetch_alpha_earth_patch(roi: ee.Geometry, start_date: str, end_date: str):
    """
    PLANNED: Fetch AlphaEarth Foundation Model embedding patch for a given ROI.

    AlphaEarth produces dense geospatial embeddings (similar to CLIP image embeddings
    but trained on satellite data). These embeddings capture land-cover semantics that
    raw spectral indices cannot — enabling zero-shot classification and retrieval.

    Status: NOT YET AVAILABLE in public Earth Engine datasets (as of Feb 2026).
    This function will be implemented once the dataset collection ID is published.

    Expected usage (once available):
        embeddings = ee.ImageCollection("GOOGLE/ALPHAEARTH/V1")
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                        .first()
    """
    logger.warning(
        "AlphaEarth Foundation embeddings are not yet publicly available via GEE. "
        "Using Sentinel-2 L2A bands as a substitute. "
        "This function will be activated once the collection is released."
    )
    # Fallback: return Sentinel-2 composite as proxy
    return get_sentinel2_composite(roi, start_date, end_date)
