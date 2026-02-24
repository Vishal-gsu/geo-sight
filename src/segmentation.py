"""
segmentation.py

Production integration of Meta's SAM 2 (Segment Anything Model 2) via the
`samgeo` library for zero-shot, automatic geospatial segmentation of satellite
image patches.

Workflow:
  1. Load a preprocessed RGB GeoTIFF patch.
  2. Run SAM 2 in 'everything' mode (no prompts needed) to auto-detect all objects.
  3. Post-process masks (morphological cleanup).
  4. Map segments to land-cover classes using NDVI/NDBI thresholds.
  5. Export a labelled GeoTIFF and a color overlay PNG.
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Land-cover class label map
CLASSES = {
    0: ("Unknown",    [128, 128, 128]),  # Gray
    1: ("Vegetation", [ 34, 139,  34]),  # Forest Green
    2: ("Water",      [ 65, 105, 225]),  # Royal Blue
    3: ("Urban",      [255, 140,   0]),  # Orange
    4: ("Bare Land",  [210, 180, 140]),  # Tan
}


def get_device() -> str:
    """Returns 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_sam_model(checkpoint: str = "vit_b"):
    """
    Initialises the SAM 2 model using samgeo's SamGeo wrapper.

    samgeo auto-downloads the checkpoint the first time if not found locally.

    Args:
        checkpoint: SAM 2 architecture variant. Options:
                    'vit_b' (base, balanced)  ← default — best for CPU
                    'vit_l' (large, higher quality)
                    'vit_h' (huge, best quality, needs GPU)

    Returns:
        SamGeo model instance
    """
    try:
        from samgeo import SamGeo
        device = get_device()
        logger.info(f"Initialising SAM model '{checkpoint}' on {device}")
        model = SamGeo(
            model_type=checkpoint,
            device=device,
            automatic=True    # enables auto-everything segmentation
        )
        logger.info("✅ SAM model loaded")
        return model
    except ImportError:
        logger.error("samgeo not installed. Run: pip install segment-geospatial")
        raise


def run_automatic_segmentation(
    model,
    image_path: str,
    output_dir: str = "results",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 100,
) -> tuple[str, str]:
    """
    Runs SAM 2 automatic segmentation on a GeoTIFF image.

    SAM 2 generates candidate masks for the entire image automatically. Key params:
      - points_per_side: grid density of prompt points (higher → more masks, slower)
      - pred_iou_thresh: filter out low-quality masks (0-1, higher → stricter)
      - stability_score_thresh: filter out unstable masks (0-1)
      - min_mask_region_area: remove tiny masks below N pixels

    Args:
        model       : SamGeo model instance from init_sam_model()
        image_path  : path to input GeoTIFF or PNG
        output_dir  : directory to save outputs

    Returns:
        mask_path   : path to output GeoTIFF with integer segment IDs
        overlay_path: path to PNG color overlay
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image_name = Path(image_path).stem
    mask_path    = str(output_dir / f"{image_name}_sam2_masks.tif")
    overlay_path = str(output_dir / f"{image_name}_sam2_overlay.png")

    logger.info(f"Running SAM 2 automatic segmentation on: {image_path}")
    logger.info(f"  params: points_per_side={points_per_side}, iou_thresh={pred_iou_thresh}")

    model.generate(
        source=image_path,
        output=mask_path,
        foreground=True,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    # Save a visual PNG overlay of all detected segments
    model.show_anns(
        cmap="jet",
        add_boxes=False,
        output=overlay_path,
        blend=True,
    )

    logger.info(f"✅ SAM 2 masks saved to: {mask_path}")
    logger.info(f"✅ Overlay saved to:     {overlay_path}")
    return mask_path, overlay_path


def classify_segments_by_spectral_index(
    segment_mask: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    ndbi: np.ndarray,
) -> np.ndarray:
    """
    Maps SAM 2 segment IDs to land-cover class IDs using per-segment mean
    spectral index values and simple threshold rules:

      - mean NDVI > 0.35 → Vegetation (1)
      - mean NDWI > 0.10 → Water (2)
      - mean NDBI > 0.05 → Urban (3)
      - mean NDVI between 0.05 and 0.35 and none of above → Bare Land (4)
      - else → Unknown (0)

    Args:
        segment_mask : (H, W) int16 — SAM 2 output with integer segment IDs
        ndvi, ndwi, ndbi : (H, W) float — spectral index arrays

    Returns:
        class_map : (H, W) uint8 — pixel class labels per CLASSES dict
    """
    logger.info("Classifying SAM 2 segments using spectral indices…")
    class_map = np.zeros(segment_mask.shape, dtype=np.uint8)  # default = Unknown

    unique_ids = np.unique(segment_mask)
    unique_ids = unique_ids[unique_ids > 0]  # skip background (0)

    for seg_id in unique_ids:
        mask = segment_mask == seg_id
        mean_ndvi = float(np.nanmean(ndvi[mask]))
        mean_ndwi = float(np.nanmean(ndwi[mask]))
        mean_ndbi = float(np.nanmean(ndbi[mask]))

        if mean_ndvi > 0.35:
            label = 1      # Vegetation
        elif mean_ndwi > 0.10:
            label = 2      # Water
        elif mean_ndbi > 0.05:
            label = 3      # Urban
        elif mean_ndvi > 0.05:
            label = 4      # Bare Land
        else:
            label = 0      # Unknown

        class_map[mask] = label

    unique_classes, counts = np.unique(class_map, return_counts=True)
    total = class_map.size
    logger.info("Segment classification results:")
    for cls, cnt in zip(unique_classes, counts):
        name = CLASSES.get(int(cls), ("Unknown", None))[0]
        logger.info(f"  {name:12s}: {cnt:6d} px ({cnt/total*100:5.1f}%)")

    return class_map


def apply_morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = 5,
    operation: str = "opening"
) -> np.ndarray:
    """
    Cleans up a binary or multi-class segmentation mask using morphological operations.

    Args:
        mask         : (H, W) np.ndarray
        kernel_size  : size of the structuring element (square kernel)
        operation    : 'opening'  (remove small noise specks)
                       'closing'  (fill small internal holes)
                       'both'     (opening then closing)

    Returns:
        cleaned_mask : (H, W) np.ndarray  same dtype as input
    """
    logger.info(f"Applying morphological {operation} (kernel={kernel_size}×{kernel_size})")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = mask.copy()

    if operation in ("opening", "both"):
        result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN,  kernel)
    if operation in ("closing", "both"):
        result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return result.astype(mask.dtype)


def render_class_overlay(
    rgb_image: np.ndarray,
    class_map: np.ndarray,
    alpha: float = 0.45,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Blends a class map colour overlay onto an RGB image for visualization.

    Args:
        rgb_image   : (H, W, 3) uint8
        class_map   : (H, W) uint8
        alpha       : overlay transparency (0=invisible, 1=opaque)
        output_path : if given, saves the result as PNG

    Returns:
        overlay : (H, W, 3) uint8
    """
    colour_layer = np.zeros_like(rgb_image)
    for class_id, (name, colour) in CLASSES.items():
        colour_layer[class_map == class_id] = colour

    overlay = (rgb_image * (1 - alpha) + colour_layer * alpha).astype(np.uint8)

    if output_path:
        from PIL import Image
        Image.fromarray(overlay).save(output_path)
        logger.info(f"✅ Class overlay saved: {output_path}")

    return overlay
