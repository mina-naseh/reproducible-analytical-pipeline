import os
import shutil
import logging
import geopandas as gpd
from src.pre_visualization import (
    inspect_geojson_data,
    plot_geojson_species_map,
    inspect_all_las_files,
    inspect_las_data,
    plot_tree_type_distribution_by_plot,
    plot_species_distribution_for_all_plots,
    plot_field_density,
    visualize_point_clouds_with_colorbar,
    preprocess_las_for_models
)
from src.lmf_utils import process_all_las_files_with_ground_truth

# --- Global Configuration ---
RESULTS_DIR = "./results"
PRE_VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "pre_visualization")
LMF_DIR = os.path.join(RESULTS_DIR, "lmf")

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
    """Ensures directory exists, clearing contents if specified."""
    if os.path.exists(directory):
        if clear:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(directory)

def main():
    """Main function to process data for both visualization and Local Maxima Filtering."""
    logger.info("Setting up results directories...")
    setup_directory(RESULTS_DIR, clear=False)
    setup_directory(PRE_VISUALIZATION_DIR, clear=False)
    setup_directory(LMF_DIR, clear=True)

    # --- Step 1: Pre-visualization workflow ---
    logger.info("Starting data pre-visualization workflow...")

    os.makedirs(PRE_VISUALIZATION_DIR, exist_ok=True)

    field_survey = gpd.read_file(FIELD_SURVEY_PATH)
    inspect_geojson_data(field_survey, PRE_VISUALIZATION_DIR)
    plot_geojson_species_map(field_survey, save_path=os.path.join(PRE_VISUALIZATION_DIR, "species_map.png"))

    las_dir = os.path.join(DATA_DIR, "als")
    inspect_all_las_files(las_dir, os.path.join(PRE_VISUALIZATION_DIR, "las_details"))
    inspect_las_data(las_dir, save_path=PRE_VISUALIZATION_DIR)

    plot_tree_type_distribution_by_plot(
        gdf=field_survey,
        save_path=os.path.join(PRE_VISUALIZATION_DIR, "tree_type_distribution_by_plot.png"),
    )

    plot_species_distribution_for_all_plots(
        gdf=field_survey,
        output_dir=os.path.join(PRE_VISUALIZATION_DIR, "species_distribution_plots"),
    )

    plot_field_density(
        df=field_survey,
        columns=field_survey.select_dtypes(include=["number"]).columns.tolist(),
        save_path=os.path.join(PRE_VISUALIZATION_DIR, "field_density_plots.png"),
    )

    visualize_point_clouds_with_colorbar(
        las_dir=os.path.join(DATA_DIR, "als"),
        output_dir=os.path.join(PRE_VISUALIZATION_DIR, "point_cloud_plots"),
    )

    # --- Step 2: Preprocessing LAS Files for Models ---
    logger.info("Preprocessing LAS files for LMF and Point Transformer...")
    
    os.makedirs(ALS_PREPROCESSED_DIR, exist_ok=True)
    
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    for las_file in las_files:
        preprocess_las_for_models(las_file, base_dir=ALS_PREPROCESSED_DIR, height_threshold=2.0, output_dir=PRE_VISUALIZATION_DIR)

    logger.info("Pre-visualization workflow complete!")

    # --- Step 3: Local Maxima Filtering Pipeline ---
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
