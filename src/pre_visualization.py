import matplotlib.pyplot as plt
import logging
import seaborn as sns
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
# import rasterio
import json
import scipy.interpolate
import csv

LAS_GROUND_CLASS = 2

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --- Utility Functions ---
def save_plot(save_path):
    """
    Saves the current plot to a file.

    Parameters:
    - save_path (str): Path to save the plot.
    """
    plt.savefig(save_path, bbox_inches="tight")
    logger.info(f"Plot saved to {save_path}")
    plt.close()

def inspect_geojson_data(gdf, save_path):
    """
    Inspects GeoJSON data for label distributions and missing values,
    saves the report, and visualizes species distribution.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to inspect.
    - save_path (str): Path to save the summary reports and visualizations.
    """
    logger.info("Inspecting GeoJSON data...")

    os.makedirs(save_path, exist_ok=True)

    species_counts = gdf["species"].value_counts().reset_index()
    species_counts.columns = ["species", "count"]
    species_counts["Type"] = species_counts["species"].apply(
        lambda x: "Coniferous" if x in ["Fir", "Pine", "Spruce"] else "Deciduous"
    )
    species_counts.to_csv(os.path.join(save_path, "species_distribution.csv"), index=False)
    logger.info("Species distribution saved as species_distribution.csv")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=species_counts,
        x="count",
        y="species",
        hue="Type",
        dodge=False,
        hue_order=["Coniferous", "Deciduous"],
        palette=["#2A5C03", "#DAA520"],
        saturation=1,
    )
    ax.set_title("Species Distribution in the Dataset", fontsize=16)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Species", fontsize=12)
    ax.grid(axis="x", color="black", alpha=0.1)

    plot_path = os.path.join(save_path, "species_distribution.png")
    save_plot(plot_path)


# --- LAS Functions ---
def inspect_las_file(las_file, output_dir):
    """
    Inspects a LAS file, including header and point attributes, and saves the output.

    Parameters:
    - las_file (str): Path to the LAS file.
    - output_dir (str): Directory to save detailed inspection outputs.
    """
    las = laspy.read(las_file)
    header_info = {
        "version": las.header.version,
        "system_identifier": las.header.system_identifier,
        "generating_software": las.header.generating_software,
        "point_count": int(las.header.point_count),
        "bounds": {
            "x": [float(las.x.min()), float(las.x.max())],
            "y": [float(las.y.min()), float(las.y.max())],
            "z": [float(las.z.min()), float(las.z.max())],
        },
        "point_format": int(las.header.point_format.id),
        "scales": [float(scale) for scale in las.header.scales],
        "offsets": [float(offset) for offset in las.header.offsets],
    }

    point_attributes = {
        dim: list(map(float, getattr(las, dim)[:10])) for dim in las.point_format.dimension_names
    }

    output = {
        "header": header_info,
        "point_attributes": point_attributes,
    }

    output_file = os.path.join(output_dir, f"{os.path.basename(las_file)}_details.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    logger.info(f"Saved detailed inspection for {os.path.basename(las_file)} to {output_file}")


def inspect_all_las_files(las_dir, output_dir):
    """
    Inspects all LAS files in a directory and saves detailed outputs.

    Parameters:
    - las_dir (str): Directory containing LAS files.
    - output_dir (str): Directory to save detailed inspection outputs.
    """
    logger.info("Inspecting all LAS files...")
    os.makedirs(output_dir, exist_ok=True)

    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    for las_file in las_files:
        inspect_las_file(las_file, output_dir)


def inspect_las_data(las_dir, save_path):
    """
    Aggregates summary statistics for all LAS files and saves them to a CSV.

    Parameters:
    - las_dir (str): Directory containing LAS files.
    - save_path (str): Path to save the summary report.
    """
    logger.info("Aggregating LAS file statistics...")
    os.makedirs(save_path, exist_ok=True)

    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    stats = []

    for file in las_files:
        las = laspy.read(file)
        points = np.vstack((las.x, las.y, las.z)).T
        stats.append({
            "file": os.path.basename(file),
            "num_points": len(points),
            "min_x": points[:, 0].min(),
            "max_x": points[:, 0].max(),
            "min_y": points[:, 1].min(),
            "max_y": points[:, 1].max(),
            "min_z": points[:, 2].min(),
            "max_z": points[:, 2].max(),
            "unique_classes": len(np.unique(getattr(las, "classification", []))),
        })

    stats_df = pd.DataFrame(stats)
    stats_file = os.path.join(save_path, "las_stats.csv")
    stats_df.to_csv(stats_file, index=False)
    logger.info(f"Saved LAS file statistics to {stats_file}")


# --- Visualization Functions ---
def plot_geojson_species_map(gdf, save_path):
    """
    Plots a geographic map of trees colored by species and saves the plot.
    """
    gdf = gdf.to_crs(epsg=4326)
    ax = gdf.plot(column="species", cmap="viridis", legend=True, figsize=(15, 15), markersize=1)
    plt.title("Tree Locations Colored by Species", fontsize=18)
    save_plot(save_path)

# def visualize_raster_images(tif_dir, save_path):
#     """
#     Visualizes raster images in a 5x2 layout and saves the plot.

#     Parameters:
#     - tif_dir (str): Path to the directory containing raster files (.tif).
#     - save_path (str): Path to save the combined plot.
#     """
#     logger.info(f"Visualizing raster images in {tif_dir}...")

#     tiff_files = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith(".tif")]
#     if not tiff_files:
#         logger.warning(f"No raster files found in {tif_dir}.")
#         return
    
#     num_files = len(tiff_files)
#     assert num_files <= 10, "The directory contains more than 10 raster files."

#     num_cols = 5
#     num_rows = 2

#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

#     axes = axes.flatten()

#     for i, (tif_file, ax) in enumerate(zip(tiff_files, axes)):
#         with rasterio.open(tif_file) as src:
#             img = src.read()
#             ax.imshow(np.rollaxis(img, 0, 3) / 255.0)
#             ax.set_title(f"Raster: {os.path.basename(tif_file)}", fontsize=12)
#             ax.axis("off")

#     for ax in axes[len(tiff_files):]:
#         ax.set_visible(False)

#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight")
#     logger.info(f"Raster images saved in 5x2 layout as {save_path}")
#     plt.close()


def plot_tree_type_distribution_by_plot(gdf, save_path):
    """
    Plots the distribution of coniferous and deciduous trees across plots and saves the plot.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame containing tree data.
    - save_path (str): Path to save the plot image.
    """
    logger.info("Visualizing tree type distribution by plot...")

    conifers = ["Fir", "Pine", "Spruce"] 
    gdf["Type"] = gdf["species"].map(lambda x: "Coniferous" if x in conifers else "Deciduous")

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=gdf,
        y="plot",
        discrete=True,
        hue="Type",
        hue_order=["Coniferous", "Deciduous"],
        palette=["#2A5C03", "#DAA520"],
        multiple="stack",
        shrink=0.75,
        alpha=1,
        lw=0,
    )

    ax.set_ylim(gdf["plot"].max() + 0.5, gdf["plot"].min() - 0.5)
    ax.set_ylabel("Plot", fontsize=12)
    ax.set_xlabel("Tree Count", fontsize=12)
    ax.set_title("Tree Type Distribution by Plot", fontsize=16)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    ax.grid(axis="x", color="black", alpha=0.1)
    ax.grid(axis="x", which="minor", color="black", alpha=0.1)

    save_plot(save_path)


def plot_species_distribution_for_all_plots(gdf, output_dir):
    """
    Plots the spatial distribution of species for all plots and saves each plot with a corresponding name.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame containing tree data.
    - output_dir (str): Directory to save the plot images.
    """
    logger.info("Visualizing species distribution for all plots...")

    os.makedirs(output_dir, exist_ok=True)

    unique_plots = gdf["plot"].unique()
    for plot_number in unique_plots:
        logger.info(f"Processing plot {plot_number}...")

        plot_data = gdf.query("plot == @plot_number")

        if plot_data.empty:
            logger.warning(f"No data available for plot {plot_number}. Skipping.")
            continue

        ax = plot_data.plot(
            column="species",
            legend=True,
            s=5,
            aspect="equal",
            figsize=(8, 8),
            cmap="viridis",
        )
        ax.set_title(f"Species Distribution in Plot {int(plot_number)}", fontsize=16)

        save_path = os.path.join(output_dir, f"species_distribution_plot_{int(plot_number)}.png")
        save_plot(save_path)


def visualize_point_clouds_with_colorbar(las_dir, output_dir):
    """
    Generates 3D scatter plots for LAS files with a color bar for height and saves them.

    Parameters:
    - las_dir (str): Path to the directory containing LAS files.
    - output_dir (str): Path to save the plots.
    """
    logger.info("Visualizing 3D point clouds with height color bar...")
    os.makedirs(output_dir, exist_ok=True)

    las_files = sorted([os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")])
    for las_file in las_files:
        plot_name = os.path.basename(las_file).split(".")[0]
        logger.info(f"Processing {plot_name}...")

        cloud = laspy.read(las_file).xyz
        cloud -= cloud.min(axis=0, keepdims=True)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection="3d")
        scatter = ax.scatter(*cloud.swapaxes(0, 1), c=cloud[:, 2], cmap="viridis", s=2)
        ax.view_init(elev=30, azim=0)
        ax.set_title(f"{plot_name}", y=0.85)
        ax.set_aspect("auto")

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label("Height (Z-axis)", fontsize=12)

        save_path = os.path.join(output_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved 3D plot for {plot_name} with color bar as {save_path}")


def plot_field_density(df, columns, save_path):
    """
    Creates density plots for specified columns and saves the plot.

    Parameters:
    - df (DataFrame): The DataFrame containing the columns to plot.
    - columns (list): List of column names to plot density distributions for.
    - save_path (str): Path to save the combined plot.
    """
    if df.empty:
        logger.warning("The DataFrame is empty. Skipping density plot.")
        return

    logger.info(f"Creating density plots for columns: {columns}")
    num_cols = len(columns)
    plt.figure(figsize=(num_cols * 6, 5))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, num_cols, i)
        sns.kdeplot(df[col], fill=True)
        plt.title(f"Density Plot for {col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Density", fontsize=12)

    plt.tight_layout()
    save_plot(save_path)


# --- normalizing and filtering Functions ---
def normalize_cloud_height(las):
    """
    Normalizes the height of a point cloud using the ground classification.

    Parameters:
    - las: laspy.LasData object representing the LAS file.

    Returns:
    - numpy.ndarray: Normalized point cloud with heights relative to the ground.
    """
    out = las.xyz.copy()
    ground_mask = las.classification == LAS_GROUND_CLASS

    assert np.any(ground_mask), "No ground points found in the LAS file!"

    ground_level = scipy.interpolate.griddata(
        points=las.xyz[ground_mask, :2],
        values=las.xyz[ground_mask, 2],
        xi=las.xyz[:, :2],
        method="nearest",
    )
    out[:, 2] -= ground_level
    return out


def preprocess_las_for_models(las_file, base_dir="./data/als_preprocessed", height_threshold=2.0, output_dir="./data/output"):
    """
    Preprocesses a LAS file for LMF and Point Transformer models and logs point statistics.

    Parameters:
    - las_file (str): Path to the LAS file.
    - base_dir (str): Base directory to save preprocessed data.
    - height_threshold (float): Minimum height to include (meters).
    - output_dir (str): Directory to save the preprocessing report.
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    report_file = os.path.join(output_dir, "preprocessing_report.csv")
    if not os.path.exists(report_file):
        with open(report_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["File", "Vegetation Points Before Threshold", "Vegetation Points After Threshold", "Points Removed"])

    las = laspy.read(las_file)

    # Normalize height
    normalized_points = normalize_cloud_height(las)

    # Filter ground and vegetation points
    ground_mask = las.classification == LAS_GROUND_CLASS
    ground_points = las.xyz[ground_mask]
    vegetation_points = normalized_points[~ground_mask]

    total_points_before = len(vegetation_points)
    vegetation_points = vegetation_points[vegetation_points[:, 2] >= height_threshold]
    total_points_after = len(vegetation_points)
    points_removed = total_points_before - total_points_after

    las_name = os.path.basename(las_file).replace(".las", "")
    output_dir = os.path.join(base_dir, las_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save normalized vegetation points
    vegetation_file = os.path.join(output_dir, f"{las_name}_vegetation.npy")
    np.save(vegetation_file, vegetation_points)
    logger.info(f"Saved filtered vegetation points (height ≥ {height_threshold}m) to {vegetation_file}")

    # Save ground points for reference
    ground_file = os.path.join(output_dir, f"{las_name}_ground.npy")
    np.save(ground_file, ground_points)
    logger.info(f"Saved ground points to {ground_file}")

    with open(report_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([las_name, total_points_before, total_points_after, points_removed])
    logger.info(f"Added report entry for {las_name}")


