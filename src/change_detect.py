"""
change_detect.py

Module for computing differences between multi-temporal analyses to detect 
urban expansion or deforestation.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_difference(image_t1: np.ndarray, image_t2: np.ndarray) -> np.ndarray:
    """Computes basic pixel-level difference between two normalized images or masks."""
    logger.info("Computing signal difference between T1 and T2.")
    if image_t1.shape != image_t2.shape:
        raise ValueError(f"Both images must have the same spatial dimensions for change detection. Got {image_t1.shape} vs {image_t2.shape}.")
    return image_t2 - image_t1

def identify_deforestation(ndvi_t1: np.ndarray, ndvi_t2: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Identifies deforestation by finding areas where NDVI dropped significantly 
    between T1 and T2. Returns a binary mask.
    """
    logger.info("Identifying probable deforestation regions.")
    # A negative change greater in magnitude than the threshold implies vegetation loss
    difference = compute_difference(ndvi_t1, ndvi_t2)
    deforestation_mask = difference < -threshold
    return deforestation_mask.astype(np.uint8)

def identify_urbanization(ndbi_t1: np.ndarray, ndbi_t2: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Identifies urban expansion by finding areas where NDBI increased significantly.
    """
    logger.info("Identifying probable urbanization regions.")
    difference = compute_difference(ndbi_t1, ndbi_t2)
    urbanization_mask = difference > threshold
    return urbanization_mask.astype(np.uint8)
