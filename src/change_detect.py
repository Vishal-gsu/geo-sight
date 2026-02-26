"""
change_detect.py

Module for multi-temporal change detection between two satellite image analyses.
Identifies and quantifies urban expansion, deforestation, and water-body change.

Core approach:
    1. Compute a spectral index (NDVI, NDBI, NDWI) at two time points T1 and T2
    2. Compute pixel-wise difference:  delta = index_T2 - index_T1
    3. Apply thresholds to classify pixels as: gain / loss / stable
    4. Quantify area of each change class in km²

Important: Always compare same-season annual composites (e.g. full-year median
           2019 vs. full-year median 2024) to avoid phenological false positives
           (seasonal NDVI variation is NOT deforestation).
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Pixel area constant ───────────────────────────────────────────────────────
# Sentinel-2 at 10m resolution: each pixel = 10m × 10m = 100 m² = 0.0001 km²
S2_PIXEL_AREA_KM2 = 0.0001


def compute_difference(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
) -> np.ndarray:
    """
    Computes pixel-wise difference between two spatially aligned index arrays.

    delta = T2 - T1
        Positive values → index increased (e.g. vegetation gain, urban growth)
        Negative values → index decreased (e.g. vegetation loss, water shrinkage)

    Args:
        image_t1, image_t2 : (H, W) float arrays — must have identical shape

    Returns:
        delta : (H, W) float array in same range as inputs
    """
    if image_t1.shape != image_t2.shape:
        raise ValueError(
            f"Shape mismatch: T1={image_t1.shape} vs T2={image_t2.shape}. "
            "Ensure both composites cover the same ROI at the same resolution."
        )
    logger.info("Computing pixel-wise difference (T2 - T1)")
    return image_t2 - image_t1


def classify_change(
    delta: np.ndarray,
    gain_threshold: float = 0.1,
    loss_threshold: float = -0.1,
) -> np.ndarray:
    """
    Classifies each pixel's change into: Loss / Stable / Gain.

    Returns:
        change_class : (H, W) int8 array
            -1 = significant loss  (delta < loss_threshold)
             0 = stable            (loss_threshold ≤ delta ≤ gain_threshold)
            +1 = significant gain  (delta > gain_threshold)

    Args:
        delta          : output of compute_difference()
        gain_threshold : positive delta threshold for "significant gain"
        loss_threshold : negative delta threshold for "significant loss"

    Typical threshold values:
        NDVI change: ±0.05 (subtle) to ±0.2 (major change)
        NDBI change: ±0.05 (subtle) to ±0.15 (major urbanization)
    """
    logger.info(f"Classifying change: loss<{loss_threshold}, gain>{gain_threshold}")
    change_class = np.zeros(delta.shape, dtype=np.int8)
    change_class[delta > gain_threshold]  = 1   # Gain
    change_class[delta < loss_threshold]  = -1  # Loss
    return change_class


def identify_deforestation(
    ndvi_t1: np.ndarray,
    ndvi_t2: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """
    Identifies likely deforestation by detecting significant NDVI decline.

    A pixel is flagged as deforested if:
        NDVI_T2 - NDVI_T1 < -threshold

    Note: A threshold of 0.2 is conservative (catches major events only).
          Use 0.1 for detecting gradual degradation.
          Combine with NDBI increase at same pixel to confirm urban conversion.

    Returns:
        binary mask (H, W) uint8: 1 = deforested, 0 = unchanged
    """
    logger.info(f"Identifying deforestation (NDVI drop > {threshold})")
    delta = compute_difference(ndvi_t1, ndvi_t2)
    return (delta < -threshold).astype(np.uint8)


def identify_urbanization(
    ndbi_t1: np.ndarray,
    ndbi_t2: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Identifies likely urban expansion by detecting significant NDBI increase.

    A pixel is flagged as newly urbanized if:
        NDBI_T2 - NDBI_T1 > threshold

    For confidence: combine with NDVI decrease at the same pixel
    (vegetation-to-urban conversion = NDVI falls AND NDBI rises simultaneously).

    Returns:
        binary mask (H, W) uint8: 1 = newly urbanized, 0 = unchanged
    """
    logger.info(f"Identifying urbanization (NDBI rise > {threshold})")
    delta = compute_difference(ndbi_t1, ndbi_t2)
    return (delta > threshold).astype(np.uint8)


def identify_water_change(
    ndwi_t1: np.ndarray,
    ndwi_t2: np.ndarray,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies water body gain and loss between two time points.

    Returns:
        water_gain : (H, W) uint8 — pixels that became water (NDWI increased > threshold)
        water_loss : (H, W) uint8 — pixels that lost water  (NDWI decreased < -threshold)
    """
    logger.info(f"Identifying water change (threshold={threshold})")
    delta = compute_difference(ndwi_t1, ndwi_t2)
    water_gain = (delta >  threshold).astype(np.uint8)
    water_loss = (delta < -threshold).astype(np.uint8)
    return water_gain, water_loss


def combined_change_map(
    ndvi_t1: np.ndarray, ndvi_t2: np.ndarray,
    ndbi_t1: np.ndarray, ndbi_t2: np.ndarray,
    ndvi_thresh: float = 0.15,
    ndbi_thresh: float = 0.08,
) -> np.ndarray:
    """
    High-confidence urban expansion detector using DUAL-INDEX confirmation.

    A pixel is classified as 'vegetation → urban conversion' only if:
        - NDVI decreased by > ndvi_thresh (vegetation lost)   AND
        - NDBI increased by > ndbi_thresh (built-up increased)

    This dual-index check dramatically reduces false positives vs. using
    NDVI alone (which could flag drought-caused die-off as urban expansion).

    Returns:
        (H, W) uint8 map:
            0 = no significant change
            1 = vegetation loss only (possible degradation / drought)
            2 = urban gain only (possible bare soil)
            3 = confirmed vegetation-to-urban conversion
    """
    logger.info("Building combined NDVI+NDBI change map")
    veg_loss  = identify_deforestation(ndvi_t1, ndvi_t2, threshold=ndvi_thresh)
    urb_gain  = identify_urbanization(ndbi_t1, ndbi_t2,  threshold=ndbi_thresh)

    result = np.zeros(ndvi_t1.shape, dtype=np.uint8)
    result[veg_loss  == 1]                              = 1   # Veg loss only
    result[urb_gain  == 1]                              = 2   # Urban gain only
    result[(veg_loss == 1) & (urb_gain == 1)]           = 3   # Confirmed conversion

    counts = {0: "Stable", 1: "Veg loss only", 2: "Urban gain only", 3: "Confirmed conversion"}
    for cls_id, name in counts.items():
        n = int((result == cls_id).sum())
        logger.info(f"  {name:25s}: {n:7d} px  ({n * S2_PIXEL_AREA_KM2:.2f} km²)")
    return result


def compute_change_statistics(
    delta: np.ndarray,
    pixel_area_km2: float = S2_PIXEL_AREA_KM2,
) -> dict:
    """
    Computes summary statistics for a delta (change) image.

    Returns a dict with:
        mean_delta      : Mean index change across all pixels
        std_delta       : Standard deviation (how spread out the change is)
        pixels_gained   : Count of pixels with positive change
        pixels_lost     : Count of pixels with negative change
        area_gained_km2 : Area of positive change in km²
        area_lost_km2   : Area of negative change in km²
        max_gain        : Maximum positive change (most improved pixel)
        max_loss        : Minimum delta (most degraded pixel)
    """
    valid = delta[np.isfinite(delta)]
    pos = valid[valid > 0]
    neg = valid[valid < 0]

    stats = {
        "mean_delta":       float(np.mean(valid)),
        "std_delta":        float(np.std(valid)),
        "pixels_gained":    int(len(pos)),
        "pixels_lost":      int(len(neg)),
        "area_gained_km2":  round(len(pos) * pixel_area_km2, 4),
        "area_lost_km2":    round(len(neg) * pixel_area_km2, 4),
        "max_gain":         float(np.max(valid)) if len(pos) else 0.0,
        "max_loss":         float(np.min(valid)) if len(neg) else 0.0,
    }

    logger.info("Change statistics:")
    for k, v in stats.items():
        logger.info(f"  {k:20s}: {v}")
    return stats
