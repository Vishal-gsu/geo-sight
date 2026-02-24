import os
import numpy as np
import matplotlib.pyplot as plt
from src.spectral import calculate_ndvi, calculate_ndwi
from src.change_detect import compute_difference, identify_deforestation
import logging

# Setup basic logging to see the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_mock_satellite_data(shape=(500, 500)):
    """Generates synthetic multispectral surface reflectance data for testing."""
    logger.info("Generating synthetic 500x500 multispectral satellite data...")
    # Simulate slightly different data for two timestamps
    np.random.seed(42)
    
    # R, G, B, NIR, SWIR mock data bounded [0.0, 1.0]
    data_t1 = {
        "red": np.random.uniform(0.05, 0.3, shape),
        "green": np.random.uniform(0.05, 0.4, shape),
        "blue": np.random.uniform(0.02, 0.2, shape),
        "nir": np.random.uniform(0.1, 0.8, shape),  # High NIR = Vegetation
        "swir": np.random.uniform(0.05, 0.5, shape)
    }
    
    # Introduce "deforestation" in the center for T2
    data_t2 = {k: v.copy() for k, v in data_t1.items()}
    # Drop NIR and increase Red/SWIR in the center to simulate lost vegetation -> bare soil/urban
    center_y, center_x = shape[0]//2, shape[1]//2
    data_t2["nir"][center_y-50:center_y+50, center_x-50:center_x+50] *= 0.2
    data_t2["red"][center_y-50:center_y+50, center_x-50:center_x+50] += 0.2
    data_t2["swir"][center_y-50:center_y+50, center_x-50:center_x+50] += 0.2
    
    return data_t1, data_t2

def plot_results(ndvi_t1, ndvi_t2, deforestation_mask):
    """Saves a demonstration plot of the pipeline outputs."""
    logger.info("Plotting analysis results to results/demo_output.png")
    os.makedirs("results", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot T1
    im1 = axes[0].imshow(ndvi_t1, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0].set_title('NDVI (Time 1)')
    axes[0].axis('off')
    
    # Plot T2
    im2 = axes[1].imshow(ndvi_t2, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title('NDVI (Time 2 - Deforestation Added)')
    axes[1].axis('off')
    
    # Plot Mask
    im3 = axes[2].imshow(deforestation_mask, cmap='Reds', interpolation='none')
    axes[2].set_title('Identified Deforestation Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/demo_output.png", dpi=150)
    logger.info("Successfully saved results/demo_output.png")

def main():
    logger.info("🚀 Starting GeoSight Local Demonstration Pipeline")
    
    # 1. Load (Mock) Data
    data_t1, data_t2 = generate_mock_satellite_data()
    
    # 2. Spectral Analysis (NDVI)
    logger.info("Running Spectral Analysis Module (NDVI)...")
    ndvi_t1 = calculate_ndvi(data_t1["nir"], data_t1["red"])
    ndvi_t2 = calculate_ndvi(data_t2["nir"], data_t2["red"])
    
    # 3. Change Detection
    logger.info("Running Change Detection Module...")
    # We define deforestation as an NDVI drop > 0.3
    deforestation_mask = identify_deforestation(ndvi_t1, ndvi_t2, threshold=0.3)
    
    # 4. Report / Visualization
    plot_results(ndvi_t1, ndvi_t2, deforestation_mask)
    
    logger.info("✅ Pipeline demonstration complete!")

if __name__ == "__main__":
    main()
