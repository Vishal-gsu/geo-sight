"""
spectral.py

Mathematical definitions of spectral indices (band math) used in Earth Observation.
All functions operate on NumPy arrays of surface reflectance values in [0, 1].

Indices implemented:
    - NDVI  : Normalized Difference Vegetation Index
    - NDWI  : Normalized Difference Water Index
    - NDBI  : Normalized Difference Built-up Index
    - EVI   : Enhanced Vegetation Index  (reduces saturation + atmospheric noise)
    - SAVI  : Soil Adjusted Vegetation Index (reduces bare soil background effect)

Why 'Normalized Difference' form?
    Formula: (A - B) / (A + B)
    - Illumination-invariant: if solar angle changes, both A and B scale equally → ratio unchanged
    - Bounded output: always in [-1, 1] for non-negative reflectances
    - Amplifies contrast between materials that respond oppositely in A and B bands
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

EPS = 1e-10   # prevents division by zero without affecting results


def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)

    Physical basis:
        - Chlorophyll absorbs Red (~665 nm) for photosynthesis → low Red reflectance
        - Leaf cell walls scatter NIR (~842 nm) → high NIR reflectance
        - Healthy vegetation: NDVI ≈ 0.3–0.9
        - Bare soil: NDVI ≈ 0.0–0.1
        - Water: NDVI < 0 (NIR absorbed by water)

    Sentinel-2 bands: NIR = B8 (842 nm), Red = B4 (665 nm)

    Limitation: NDVI saturates at LAI > 3 (very dense canopies).
    Use EVI for dense tropical forest.
    """
    logger.info("Calculating NDVI")
    ndvi = (nir_band - red_band) / (nir_band + red_band + EPS)
    return np.clip(ndvi, -1.0, 1.0)


def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index (McFeeters 1996).
    NDWI = (Green - NIR) / (Green + NIR)

    Physical basis:
        - Water strongly absorbs NIR → very low NIR reflectance
        - Water reflects Green moderately
        - Open water: NDWI > 0
        - Vegetation: NDWI < 0 (high NIR)
        - Urban: NDWI < 0 (moderate NIR from building materials)

    Sentinel-2 bands: Green = B3 (560 nm), NIR = B8 (842 nm)

    Note: NDWI can be confused with built-up areas. Use in combination
    with NDVI and NDBI to disambiguate.
    """
    logger.info("Calculating NDWI")
    ndwi = (green_band - nir_band) / (green_band + nir_band + EPS)
    return np.clip(ndwi, -1.0, 1.0)


def calculate_ndbi(swir_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Built-up Index (Zha et al., 2003).
    NDBI = (SWIR - NIR) / (SWIR + NIR)

    Physical basis:
        - Concrete, asphalt, roof tiles reflect SWIR (~1614 nm) strongly
        - Vegetation absorbs SWIR (via water content in leaves)
        - Built-up: NDBI > 0
        - Vegetation: NDBI < 0
        - Water: NDBI < 0

    Sentinel-2 bands: SWIR = B11 (1614 nm), NIR = B8 (842 nm)
    """
    logger.info("Calculating NDBI")
    ndbi = (swir_band - nir_band) / (swir_band + nir_band + EPS)
    return np.clip(ndbi, -1.0, 1.0)


def calculate_evi(
    nir_band: np.ndarray,
    red_band: np.ndarray,
    blue_band: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
) -> np.ndarray:
    """
    Enhanced Vegetation Index (Huete et al., 2002).
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    Improvements over NDVI:
        1. Does NOT saturate in dense canopies (Leaf Area Index > 3)
        2. Reduces atmospheric aerosol effects via the Blue band correction (C2*Blue)
        3. Reduces soil background signal via the L (canopy background adjustment)

    Default coefficients are those of the standard MODIS EVI formulation.
    Range: approximately [-1, 1], typical vegetation: 0.2–0.8

    Sentinel-2 bands: NIR = B8, Red = B4, Blue = B2
    """
    logger.info("Calculating EVI")
    evi = G * (nir_band - red_band) / (
        nir_band + C1 * red_band - C2 * blue_band + L + EPS
    )
    return np.clip(evi, -1.0, 1.0)


def calculate_savi(
    nir_band: np.ndarray,
    red_band: np.ndarray,
    L: float = 0.5,
) -> np.ndarray:
    """
    Soil Adjusted Vegetation Index (Huete, 1988).
    SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L)

    Purpose:
        Reduces the influence of soil brightness variations on vegetation detection.
        NDVI is sensitive to soil background — in sparsely vegetated areas
        (arid land, newly planted fields), bright or dark soil affects NDVI heavily.
        The L factor adjusts for the soil noise:
            L = 0   : no soil correction (= NDVI)
            L = 0.5 : optimal for intermediate vegetation density (default)
            L = 1.0 : best for very sparse vegetation (<15% cover)

    Range: [-1.5, 1.5] approximately. Typical vegetation: 0.1–0.7

    Sentinel-2 bands: NIR = B8, Red = B4
    """
    logger.info(f"Calculating SAVI (L={L})")
    savi = (1.0 + L) * (nir_band - red_band) / (nir_band + red_band + L + EPS)
    return np.clip(savi, -2.0, 2.0)


def compute_all_indices(
    nir: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    swir: np.ndarray,
) -> dict:
    """
    Convenience function: computes all five indices in one call.

    Args:
        nir, red, green, blue, swir : (H, W) float32 arrays in [0, 1]

    Returns:
        dict with keys: 'ndvi', 'ndwi', 'ndbi', 'evi', 'savi'
        each value is an (H, W) float32 array
    """
    return {
        "ndvi": calculate_ndvi(nir, red),
        "ndwi": calculate_ndwi(green, nir),
        "ndbi": calculate_ndbi(swir, nir),
        "evi":  calculate_evi(nir, red, blue),
        "savi": calculate_savi(nir, red),
    }
