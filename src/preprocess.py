"""
preprocess.py

Production-level module for reading, cleaning, and normalizing raw satellite imagery.
Handles coordinate transformations, radiometric calibration, and cloud masking for
Sentinel-2 Level-2A GeoTIFF files.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sentinel-2 L2A quantification factor (reflectance = DN / 10000)
S2_QUANTIFICATION = 10_000.0

# Sentinel-2 Band → index mapping within standard export order [B2, B3, B4, B8, B11]
S2_BANDS = {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4}

# SCL cloud/shadow classes to mask
SCL_CLOUD_VALUES = {8, 9, 10}   # cloud med prob, cloud high prob, thin cirrus
SCL_SHADOW_VALUE = 3            # cloud shadow


def load_geotiff(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Loads a multi-band GeoTIFF into a float32 NumPy array.

    Returns:
        image_array : np.ndarray  shape (bands, H, W), float32, values in [0, 1]
        meta        : dict        rasterio metadata including CRS, transform
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {filepath}")

    logger.info(f"Loading raster: {filepath.name}")
    with rasterio.open(filepath) as src:
        meta = src.meta.copy()
        # Read all bands; convert from int16 to float32, apply S2 scaling
        raw = src.read().astype(np.float32)

    image = np.clip(raw / S2_QUANTIFICATION, 0.0, 1.0)
    logger.info(f"  → shape={image.shape}, dtype={image.dtype}, CRS={meta['crs']}")
    return image, meta


def normalize_reflectance(
    image_array: np.ndarray,
    clip_percentile: float = 98.0
) -> np.ndarray:
    """
    Per-band percentile clipping + Min-Max normalization to [0, 1].

    Percentile clipping prevents a single saturated pixel (cloud edge,
    metallic roof) from compressing the dynamic range of the entire image.

    Args:
        image_array   : (bands, H, W) float32 array
        clip_percentile: upper percentile used as max (e.g., 98 → ignores top 2%)

    Returns:
        normalized    : (bands, H, W) float32, values in [0, 1]
    """
    logger.info(f"Normalizing reflectance (clip at {clip_percentile}th percentile per band)")
    normalized = np.zeros_like(image_array, dtype=np.float32)

    for b in range(image_array.shape[0]):
        band = image_array[b]
        valid = band[np.isfinite(band)]  # ignore NaN cloud-masked pixels
        if valid.size == 0:
            logger.warning(f"  Band {b}: all pixels are NaN — skipping")
            continue
        p_low  = float(np.percentile(valid, 100.0 - clip_percentile))
        p_high = float(np.percentile(valid, clip_percentile))
        denom  = (p_high - p_low) if (p_high - p_low) > 1e-8 else 1.0
        normalized[b] = np.clip((band - p_low) / denom, 0.0, 1.0)
        logger.info(f"  Band {b}: [{p_low:.4f}, {p_high:.4f}] → [0, 1]")

    return normalized


def apply_cloud_mask(
    image: np.ndarray,
    scl_band: Optional[np.ndarray] = None,
    custom_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Applies a cloud/shadow mask: sets affected pixels to NaN.

    Prefers SCL band (Sentinel-2 Scene Classification Layer) when provided.
    Falls back to a custom boolean mask array (True = cloud).

    Args:
        image       : (bands, H, W) float32
        scl_band    : (H, W) int   SCL classification band from Sentinel-2 L2A
        custom_mask : (H, W) bool  True where pixel should be masked

    Returns:
        masked_image: (bands, H, W) float32 with NaN at cloud pixels
    """
    masked = image.copy()

    if scl_band is not None:
        cloud_pixels = np.isin(scl_band, list(SCL_CLOUD_VALUES | {SCL_SHADOW_VALUE}))
        n_cloudy = cloud_pixels.sum()
        logger.info(f"Applying SCL cloud mask: {n_cloudy} pixels masked ({n_cloudy/cloud_pixels.size*100:.1f}%)")
        masked[:, cloud_pixels] = np.nan
    elif custom_mask is not None:
        n_masked = custom_mask.sum()
        logger.info(f"Applying custom cloud mask: {n_masked} pixels masked")
        masked[:, custom_mask] = np.nan
    else:
        logger.warning("No cloud mask provided — returning unmasked image")

    return masked


def reproject_to_utm(image: np.ndarray, meta: dict, target_epsg: int = 32643) -> tuple[np.ndarray, dict]:
    """
    Reprojects image to a target UTM CRS (default: UTM Zone 43N for India).

    Args:
        image       : (bands, H, W)
        meta        : rasterio metadata dict
        target_epsg : EPSG code for target CRS

    Returns:
        reprojected : (bands, newH, newW)
        new_meta    : updated metadata with new CRS + transform
    """
    from rasterio.crs import CRS
    target_crs = CRS.from_epsg(target_epsg)

    if meta["crs"] == target_crs:
        logger.info(f"Already in EPSG:{target_epsg}, skipping reprojection")
        return image, meta

    logger.info(f"Reprojecting from {meta['crs']} → EPSG:{target_epsg}")
    n_bands, h, w = image.shape
    transform, new_w, new_h = calculate_default_transform(
        meta["crs"], target_crs, w, h, *rasterio.transform.array_bounds(h, w, meta["transform"])
    )
    reprojected = np.zeros((n_bands, new_h, new_w), dtype=np.float32)

    for b in range(n_bands):
        reproject(
            source=image[b],
            destination=reprojected[b],
            src_transform=meta["transform"],
            src_crs=meta["crs"],
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

    new_meta = meta.copy()
    new_meta.update({"crs": target_crs, "transform": transform, "width": new_w, "height": new_h})
    logger.info(f"  → new shape: ({n_bands}, {new_h}, {new_w})")
    return reprojected, new_meta


def get_rgb_preview(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Extracts an 8-bit RGB preview array suitable for matplotlib / PIL.
    Assumes image band order: [Blue, Green, Red, NIR, SWIR] (Sentinel-2 export order).

    Returns:
        rgb : (H, W, 3) uint8
    """
    r = image[S2_BANDS["red"]]
    g = image[S2_BANDS["green"]]
    b = image[S2_BANDS["blue"]]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1) ** (1.0 / gamma)  # optional gamma correction
    return (rgb * 255).astype(np.uint8)
