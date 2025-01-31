import os
import shutil
import logging
import geopandas as gpd
from src.pre_visualization import process_data
from src.lmf_utils import process_all_las_files_with_ground_truth

# Define a single results directory
RESULTS_DIR = "./results"
PRE_VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "pre_visualization")
LMF_DIR = os.path.join(RESULTS_DIR, "lmf")

# Data paths
DATA_DIR = "./data"
ALS_PREPROCESSED_DIR = os.path.join(DATA_DIR, "als_preprocessed")
FIELD_SURVEY_PATH = os.path.join(DATA_DIR, "field_survey.geojson")

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def setup_directory(directory, clear=False):
    """Clears a directory if it exists, but does not remove it if it's a mount."""
    if os.path.exists(directory):
        if clear:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directories inside
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(directory)

def main():
    """
    Main function to process data for both visualization and Local Maxima Filtering.
    """
    logger.info("Setting up results directories...")
    setup_directory(RESULTS_DIR, clear=False)  # Do not clear all results
    setup_directory(PRE_VISUALIZATION_DIR, clear=False)  # Avoid clearing visualization results
    setup_directory(LMF_DIR, clear=True)  # Clear only LMF results
    
    # Step 1: Pre-visualization data processing
    logger.info("Starting data pre-visualization workflow...")
    process_data(data_dir=DATA_DIR, output_dir=PRE_VISUALIZATION_DIR)
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
        save_dir=LMF_DIR,
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
