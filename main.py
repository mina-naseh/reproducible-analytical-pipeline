import os
import shutil
import logging
import geopandas as gpd
from src.pre_visualization import process_data
from src.lmf_utils import process_all_las_files_with_ground_truth

# Directories
DATA_DIR = "./data"
OUTPUT_DIR = "./results_pre"
ALS_PREPROCESSED_DIR = os.path.join(DATA_DIR, "als_preprocessed")
RESULTS_DIR = "./results_lmf"
FIELD_SURVEY_PATH = os.path.join(DATA_DIR, "field_survey.geojson")

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def setup_directory(directory):
    """
    Ensures that the given directory exists and is empty.
    
    Args:
        directory (str): Path to the directory.
    """
    if os.path.exists(directory):
        logger.info(f"Clearing existing directory: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)
    logger.info(f"Created directory: {directory}")

def main():
    """
    Main function to process data for both visualization and Local Maxima Filtering.
    """
    logger.info("Setting up output directories...")
    setup_directory(OUTPUT_DIR)
    setup_directory(RESULTS_DIR)
    
    # Step 1: Pre-visualization data processing
    logger.info("Starting data pre-visualization workflow...")
    process_data(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    logger.info("Pre-visualization workflow complete!")
    
    # Step 2: Local Maxima Filtering pipeline
    if not os.path.exists(FIELD_SURVEY_PATH):
        logger.error(f"Ground truth file not found at {FIELD_SURVEY_PATH}. Exiting...")
        return
    
    ground_truth_data = gpd.read_file(FIELD_SURVEY_PATH)
    logger.info("Starting Local Maxima Filtering pipeline...")
    metrics_summary = process_all_las_files_with_ground_truth(
        las_dir=ALS_PREPROCESSED_DIR,
        ground_truth_data=ground_truth_data,
        save_dir=RESULTS_DIR,
        max_distance=5.0,
        max_height_difference=3.0,
        window_size=2.0  
    )
    
    if metrics_summary.empty:
        logger.warning("No metrics were generated. Please check the input data.")
    else:
        logger.info("Detection metrics summary saved to results directory.")
    
    logger.info("Workflow complete!")

if __name__ == "__main__":
    main()