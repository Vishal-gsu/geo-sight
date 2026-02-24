"""
spectral.py

Module containing mathematical definitions of spectral indices (band math) 
frequently used in Earth Observation (NDVI, NDWI, NDBI).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """
    Calculates the Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)
    """
    logger.info("Calculating NDVI")
    # Adding a small epsilon to denominator to avoid division by zero
    epsilon = 1e-10
    ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)
    # Clip values mathematically bounded outside [-1, 1] due to noise
    return np.clip(ndvi, -1, 1)

def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculates the Normalized Difference Water Index (NDWI).
    NDWI = (Green - NIR) / (Green + NIR)
    """
    logger.info("Calculating NDWI")
    epsilon = 1e-10
    ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)
    return np.clip(ndwi, -1, 1)

def calculate_ndbi(swir_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculates the Normalized Difference Built-up Index (NDBI).
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """
    logger.info("Calculating NDBI")
    epsilon = 1e-10
    ndbi = (swir_band - nir_band) / (swir_band + nir_band + epsilon)
    return np.clip(ndbi, -1, 1)
