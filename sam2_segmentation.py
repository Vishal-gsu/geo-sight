"""
sam2_segmentation.py

Runs Meta's SAM 2 (via samgeo) on a downloaded satellite patch,
classifies each segment by land-cover type using spectral indices,
and exports an annotated class map, overlay visualization and stats JSON.

Usage:
    python sam2_segmentation.py                          # Bengaluru (default)
    python sam2_segmentation.py --city Mumbai
    python sam2_segmentation.py --input data/mumbai_patch/mumbai_rgb.png --city Mumbai
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--city",  default="Bengaluru",
                    help="City name, e.g. Mumbai (used for output filenames)")
parser.add_argument("--input", default=None,
                    help="Path to RGB PNG. Defaults to data/<slug>_patch/<slug>_rgb.png")
args = parser.parse_args()

CITY  = args.city
SLUG  = CITY.lower()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.segmentation import (
    init_sam_model, run_automatic_segmentation,
    classify_segments_by_spectral_index,
    apply_morphological_cleanup, render_class_overlay,
    CLASSES
)
from src.spectral import calculate_ndvi, calculate_ndwi, calculate_ndbi
from src.report import plot_classification_map, generate_class_area_report, save_stats_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
if args.input:
    RGB_PNG = Path(args.input)
else:
    RGB_PNG = Path(f"data/{SLUG}_patch/{SLUG}_rgb.png")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ── 1. Load RGB image ─────────────────────────────────────────────────────────
if not RGB_PNG.exists():
    logger.error(f"RGB patch not found: {RGB_PNG}")
    logger.error(f"Please run:  python download_tile.py --city {CITY}  first!")
    sys.exit(1)

logger.info(f"Loading satellite patch: {RGB_PNG}")
pil_img  = Image.open(RGB_PNG).convert("RGB")
rgb_arr  = np.array(pil_img)                # (H, W, 3) uint8
logger.info(f"  Patch shape: {rgb_arr.shape}")


# ── 2. Derive mock spectral indices from RGB for classification ───────────────
# Note: real NDVI needs NIR band. Here we approximate from the RGB+GEE
# for demonstration. Replace with loaded GeoTIFF bands for production.
logger.info("Computing approximate spectral proxies from RGB channels…")
r = rgb_arr[:,:,0].astype(np.float32) / 255.0
g = rgb_arr[:,:,1].astype(np.float32) / 255.0
b = rgb_arr[:,:,2].astype(np.float32) / 255.0

# These are visual proxies (not true spectral indices) from RGB only.
# For production, load B4/B8 from the GeoTIFF export instead.
epsilon = 1e-8
# Approximate "vegetation" using green vs red (Visible Atmospherically Resistant Index proxy)
ndvi_proxy = (g - r) / (g + r + epsilon)
ndwi_proxy = (g - b) / (g + b + epsilon)
ndbi_proxy = (r - g) / (r + g + epsilon)   # brighter red = potential urban


# ── 3. Run SAM 2 ──────────────────────────────────────────────────────────────
logger.info("\n=== Initialising SAM 2 ===")
try:
    sam_model = init_sam_model("vit_b")

    logger.info("\n=== Running automatic segmentation ===")
    mask_tif, overlay_png = run_automatic_segmentation(
        model=sam_model,
        image_path=str(RGB_PNG),
        output_dir=str(OUT_DIR),
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
    )

    # ── 4. Load segment IDs and classify ─────────────────────────────────────
    import rasterio
    with rasterio.open(mask_tif) as src:
        segment_mask = src.read(1).astype(np.int16)

    logger.info(f"SAM 2 detected {np.unique(segment_mask[segment_mask > 0]).size} unique segments")

    # Resize spectral proxies to match mask shape if needed
    if rgb_arr.shape[:2] != segment_mask.shape:
        from skimage.transform import resize
        ndvi_proxy = resize(ndvi_proxy, segment_mask.shape, anti_aliasing=True)
        ndwi_proxy = resize(ndwi_proxy, segment_mask.shape, anti_aliasing=True)
        ndbi_proxy = resize(ndbi_proxy, segment_mask.shape, anti_aliasing=True)

    class_map = classify_segments_by_spectral_index(segment_mask, ndvi_proxy, ndwi_proxy, ndbi_proxy)
    class_map = apply_morphological_cleanup(class_map, kernel_size=5, operation="both")

    # ── 5. Render outputs ─────────────────────────────────────────────────────
    class_overlay = render_class_overlay(
        rgb_image=rgb_arr,
        class_map=class_map,
        alpha=0.45,
        output_path=str(OUT_DIR / f"{SLUG}_class_overlay.png")
    )

    plot_classification_map(class_map, out_path=str(OUT_DIR / f"{SLUG}_classification_map.png"))

    stats = generate_class_area_report(class_map, pixel_resolution_m=3.0)
    save_stats_json(stats, str(OUT_DIR / f"{SLUG}_class_stats.json"))

    # ── 6. Summary Figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0f1117")
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#1a1d27")

    axes[0].imshow(rgb_arr);             axes[0].set_title("RGB Satellite Patch",   color="white", fontsize=11)
    axes[1].imshow(Image.open(overlay_png)); axes[1].set_title("SAM 2 Segments",   color="white", fontsize=11)
    axes[2].imshow(class_overlay);       axes[2].set_title("Land Cover Classes",    color="white", fontsize=11)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=[c/255 for c in v[1]], label=v[0]) for v in CLASSES.values() if v[0] != "Unknown"]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               facecolor="#1a1d27", labelcolor="white", fontsize=10, framealpha=0.8)

    fig.suptitle(f"GeoSight — SAM 2 Land Cover Segmentation | {CITY}, India",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_summary = str(OUT_DIR / f"{SLUG}_sam2_summary.png")
    plt.savefig(out_summary, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    logger.info("\n══════════════════════════════════════════════════")
    logger.info("  🚀 SAM 2 Segmentation Complete!")
    logger.info("══════════════════════════════════════════════════")
    for cls, s in stats.items():
        logger.info(f"  {cls:12s}: {s['area_sq_km']:6.2f} km²  ({s['percentage']:5.1f}%)")
    logger.info(f"\n  📊 Summary: {out_summary}")
    logger.info(f"  🗺️  Overlay: {OUT_DIR}/{SLUG}_class_overlay.png")
    logger.info(f"  📋 Stats:   {OUT_DIR}/{SLUG}_class_stats.json")

except Exception as e:
    logger.error(f"SAM 2 run failed: {e}")
    logger.info("\n  💡 Falling back to spectral-only classification (no SAM 2)…")

    # ── Fallback: pure spectral classification without SAM 2 ─────────────────
    H, W = ndvi_proxy.shape
    class_map_fallback = np.zeros((H, W), dtype=np.uint8)
    class_map_fallback[ndvi_proxy > 0.15] = 1   # Vegetation
    class_map_fallback[ndwi_proxy > 0.05] = 2   # Water
    class_map_fallback[ndbi_proxy > 0.08] = 3   # Urban

    class_overlay_fb = render_class_overlay(
        rgb_image=rgb_arr,
        class_map=class_map_fallback,
        alpha=0.45,
        output_path=str(OUT_DIR / f"{SLUG}_class_overlay_fallback.png")
    )
    plot_classification_map(class_map_fallback, out_path=str(OUT_DIR / f"{SLUG}_classification_fallback.png"))
    stats_fb = generate_class_area_report(class_map_fallback, pixel_resolution_m=3.0)
    save_stats_json(stats_fb, str(OUT_DIR / f"{SLUG}_class_stats_fallback.json"))

    logger.info("✅ Fallback spectral classification complete.")
    for cls, s in stats_fb.items():
        logger.info(f"  {cls:12s}: {s['area_sq_km']:6.2f} km²  ({s['percentage']:5.1f}%)")
