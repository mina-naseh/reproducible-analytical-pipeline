import os
import numpy as np
import pandas as pd
import scipy.spatial
from shapely.geometry import Point
import geopandas as gpd
import logging
import matplotlib.pyplot as plt


LAS_GROUND_CLASS = 2

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def local_maxima_filter(cloud, window_size):
    """
    Detects local maxima in the point cloud with a fixed window size.

    Parameters:
    - cloud (numpy.ndarray): Preprocessed vegetation point cloud (X, Y, Z).
    - window_size (float): Radius of the neighborhood to consider for local maxima.

    Returns:
    - numpy.ndarray: Detected tree locations and heights (X, Y, Z).
    """
    if not isinstance(cloud, np.ndarray):
        raise TypeError(f"Cloud needs to be a numpy array, not {type(cloud)}")
    if cloud.size == 0:
        logger.warning("Point cloud is empty. Returning empty array.")
        return np.array([])

    # Initialize KDTree for neighborhood queries
    tree = scipy.spatial.KDTree(data=cloud)
    seen_mask = np.zeros(cloud.shape[0], dtype=bool)
    local_maxima = []

    # Detect local maxima
    for i, point in enumerate(cloud):
        if seen_mask[i]:
            continue

        # Find neighbors within the specified window size
        neighbor_indices = tree.query_ball_point(point, window_size)
        highest_neighbor = neighbor_indices[cloud[neighbor_indices, 2].argmax()]
        seen_mask[neighbor_indices] = True
        seen_mask[highest_neighbor] = False

        if i == highest_neighbor:
            local_maxima.append(i)

    logger.info(f"Detected {len(local_maxima)} local maxima (trees).")
    return cloud[local_maxima]


def process_point_cloud_with_lmf(points, window_size):
    """
    Processes a preprocessed point cloud with local maxima filtering.

    Parameters:
    - points (numpy.ndarray): Preprocessed vegetation points (X, Y, Z).
    - window_size (float): Radius for local maxima detection.

    Returns:
    - numpy.ndarray: Detected tree locations and heights (X, Y, Z).
    """
    if points.size == 0:
        logger.warning("Point cloud is empty. Skipping processing.")
        return np.array([])

    # Apply Local Maxima Filtering
    logger.info(f"Applying Local Maxima Filtering with window_size={window_size}.")
    detected_trees = local_maxima_filter(points, window_size)
    logger.info(f"Local Maxima Filtering complete. Detected {len(detected_trees)} trees.")

    return detected_trees


def transform_ground_truth(ground_truth):
    """
    Transforms ground truth GeoDataFrame to a NumPy array for processing.

    Parameters:
    - ground_truth (GeoDataFrame): Ground truth data.

    Returns:
    - np.ndarray: Transformed ground truth data as a NumPy array.
    """
    ground_truth = ground_truth.copy()
    ground_truth["geometry.x"] = ground_truth.geometry.x
    ground_truth["geometry.y"] = ground_truth.geometry.y
    return ground_truth[["geometry.x", "geometry.y", "height"]].to_numpy()

def crop_by_other(points: np.ndarray, other: np.ndarray) -> np.ndarray:
    """
    Crop points by the convex hull of another set of points.

    Parameters:
    - points (np.ndarray): Array of points to crop (X, Y, Z, ...).
    - other (np.ndarray): Array of points defining the crop boundary (X, Y, Z, ...).

    Returns:
    - np.ndarray: Cropped points.
    """
    hull = scipy.spatial.ConvexHull(other[:, :2])  # Convex hull on the X, Y coordinates
    vertex_points = hull.points[hull.vertices]  # Extract the vertices of the hull
    delaunay = scipy.spatial.Delaunay(vertex_points)  # Create Delaunay triangulation
    within_hull = delaunay.find_simplex(points[:, :2]) >= 0  # Check points within the hull
    return points[within_hull]


def match_candidates(
    ground_truth: np.ndarray,
    candidates: np.ndarray,
    max_distance: float,
    max_height_difference: float,
) -> list[dict]:
    """
    Match ground truth trees to candidates.

    Parameters:
    - ground_truth (np.ndarray): Array of ground truth points (X, Y, Z).
    - candidates (np.ndarray): Array of candidate points (X, Y, Z).
    - max_distance (float): Maximum distance allowed for matching.
    - max_height_difference (float): Maximum height difference allowed for matching.

    Returns:
    - list[dict]: List of matches with ground truth, candidate, distance, and class (TP, FP, FN).
    """
    logger = logging.getLogger(__name__)

    if ground_truth.size == 0:
        logger.warning("No ground truth points provided.")
        return [{"ground_truth": None, "candidate": tuple(cand), "class": "FP", "distance": None} for cand in candidates]
    if candidates.size == 0:
        logger.warning("No candidate points provided.")
        return [{"ground_truth": tuple(gt), "candidate": None, "class": "FN", "distance": None} for gt in ground_truth]

    logger.info(f"Matching {len(ground_truth)} ground truth trees with {len(candidates)} candidates.")

    # Compute distance matrix
    distance_matrix = scipy.spatial.distance_matrix(ground_truth[:, :2], candidates[:, :2])
    indices = np.nonzero(distance_matrix <= max_distance)
    distances = distance_matrix[indices]
    sparse_distances = sorted((d, pair) for d, pair in zip(distances, zip(*indices)))

    ground_truth_matched_mask = np.zeros(ground_truth.shape[0], dtype=bool)
    candidates_matched_mask = np.zeros(candidates.shape[0], dtype=bool)
    matches = []

    for distance, (i, j) in sparse_distances:
        if ground_truth_matched_mask[i] or candidates_matched_mask[j]:
            continue

        height_diff = abs(ground_truth[i, 2] - candidates[j, 2])
        if np.isnan(ground_truth[i, 2]) or height_diff <= max_height_difference:
            matches.append({
                "ground_truth": tuple(ground_truth[i]),
                "candidate": tuple(candidates[j]),
                "class": "TP",
                "distance": distance,
            })
            ground_truth_matched_mask[i] = True
            candidates_matched_mask[j] = True

    matches.extend(
        {"ground_truth": tuple(ground_truth[i]), "candidate": None, "class": "FN", "distance": None}
        for i in range(len(ground_truth)) if not ground_truth_matched_mask[i]
    )

    matches.extend(
        {"ground_truth": None, "candidate": tuple(candidates[j]), "class": "FP", "distance": None}
        for j in range(len(candidates)) if not candidates_matched_mask[j]
    )

    logger.info(f"Matching complete. Total matches: {len(matches)}")
    return matches


def calculate_detection_metrics(matches):
    """
    Calculates detection metrics (precision, recall, F1-score) from match results.

    Parameters:
    - matches (list): Match results from match_candidates.

    Returns:
    - dict: Precision, recall, F1-score, and mean distance.
    """
    if not matches:
        logger.warning("No matches provided for metric calculation.")
        return {"precision": 0, "recall": 0, "f1_score": 0, "mean_distance": None}

    tp = sum(1 for m in matches if m["ground_truth"] is not None and m["candidate"] is not None)
    fp = sum(1 for m in matches if m["ground_truth"] is None and m["candidate"] is not None)
    fn = sum(1 for m in matches if m["ground_truth"] is not None and m["candidate"] is None)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    mean_distance = (
        np.mean([m["distance"] for m in matches if m["distance"] is not None])
        if tp > 0 else None
    )

    logger.info(f"Metrics calculated: TP={tp}, FP={fp}, FN={fn}")
    logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
    if mean_distance is not None:
        logger.info(f"Mean Distance: {mean_distance:.3f}")
    else:
        logger.info("Mean Distance: None (No true positives)")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_distance": mean_distance,
    }


def visualize_detection_results(
    detected_trees, ground_truth, matches, save_path=None
):
    """
    Visualizes detected trees, ground truth, and matches.

    Parameters:
    - detected_trees (np.ndarray): Detected tree locations and heights.
    - ground_truth (np.ndarray): Ground truth tree locations and heights.
    - matches (list): Match results from match_candidates.
    - save_path (str, optional): Path to save the plot.

    Returns:
    - None: Displays or saves the plot.
    """
    if detected_trees.size == 0:
        logger.warning("No detected trees to visualize.")
        return
    if ground_truth.size == 0:
        logger.warning("No ground truth trees to visualize.")
        return
    if not matches:
        logger.warning("No matches to visualize.")
        return

    logger.info(f"Visualizing {len(ground_truth)} ground truth trees and {len(detected_trees)} detected trees.")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        ground_truth[:, 0], ground_truth[:, 1],
        c="green", label="Ground Truth", s=50, alpha=0.7
    )
    ax.scatter(
        detected_trees[:, 0], detected_trees[:, 1],
        c="red", label="Detected Trees", s=50, alpha=0.7
    )

    for match in matches:
        if match["distance"] is not None:
            ax.plot(
                [match["ground_truth"][0], match["candidate"][0]],
                [match["ground_truth"][1], match["candidate"][1]],
                c="blue", linestyle="--", alpha=0.5,
            )
            ax.text(
                (match["ground_truth"][0] + match["candidate"][0]) / 2,
                (match["ground_truth"][1] + match["candidate"][1]) / 2,
                f"{match['distance']:.2f}", fontsize=8, color="blue", alpha=0.7
            )

    all_x = np.concatenate([ground_truth[:, 0], detected_trees[:, 0]])
    all_y = np.concatenate([ground_truth[:, 1], detected_trees[:, 1]])
    ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
    ax.set_ylim(all_y.min() - 10, all_y.max() + 10)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    plt.title("Tree Detection Results")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Detection results plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def get_plot_ground_truth(ground_truth_data, plot_id):
    """
    Extracts and transforms ground truth data for a specific plot.

    Parameters:
    - ground_truth_data (GeoDataFrame): Ground truth data with tree locations.
    - plot_id (int): ID of the plot.

    Returns:
    - np.ndarray: Transformed ground truth data as a NumPy array.
    """
    plot_data = ground_truth_data[ground_truth_data["plot"] == plot_id]
    if plot_data.empty:
        logger.warning(f"No ground truth data for plot {plot_id}. Skipping...")
        return np.array([])
    return transform_ground_truth(plot_data)


def detect_trees(vegetation_points, ground_points, window_size):
    """
    Detects trees using Local Maxima Filtering (LMF).

    Parameters:
    - vegetation_points (np.ndarray): Preprocessed vegetation points (X, Y, Z).
    - ground_points (np.ndarray): Preprocessed ground points (X, Y, Z).
    - window_size (float): Radius for local maxima detection.

    Returns:
    - np.ndarray: Detected tree locations and heights.
    """
    if vegetation_points.size == 0:
        logger.warning("No vegetation points provided for tree detection.")
        return np.array([])

    detected_trees = process_point_cloud_with_lmf(
        points=vegetation_points,
        window_size=window_size
    )
    logger.info(f"Detected {len(detected_trees)} trees using LMF.")
    return detected_trees


def save_detected_trees_as_geojson(detected_trees, plot_number, save_dir, crs="EPSG:32640"):
    """
    Saves detected tree locations as a GeoJSON file.

    Parameters:
    - detected_trees (np.ndarray): Detected tree locations and heights (X, Y, Z).
    - plot_number (int): Plot number for naming the GeoJSON file.
    - save_dir (str): Directory to save the GeoJSON file.
    - crs (str): Coordinate Reference System for the GeoJSON file (default: EPSG:32640).

    Returns:
    - None
    """
    if detected_trees.size == 0:
        logger.warning(f"No detected trees for plot {plot_number}. Skipping GeoJSON saving...")
        return

    detected_trees_gdf = gpd.GeoDataFrame(
        data={"height": detected_trees[:, 2]},
        geometry=gpd.points_from_xy(detected_trees[:, 0], detected_trees[:, 1]),
        crs=crs,
    )

    geojson_path = os.path.join(save_dir, f"{plot_number}_detected_trees.geojson")
    detected_trees_gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info(f"Detected trees saved to {geojson_path}")


def plot_point_cloud_with_detected_trees(non_ground_points, detected_trees, plot_name, save_dir):
    """
    Plots the point cloud with detected trees overlaid.

    Parameters:
    - non_ground_points (np.ndarray): Vegetation point cloud (X, Y, Z).
    - detected_trees (np.ndarray): Detected tree locations and heights (X, Y, Z).
    - plot_name (str): Name of the plot for saving.
    - save_dir (str): Directory to save the plot.

    Returns:
    - None
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        non_ground_points[:, 0],
        non_ground_points[:, 1],
        c=non_ground_points[:, 2],
        s=1,
        cmap="viridis",
        label="Point Cloud",
    )
    ax.scatter(
        detected_trees[:, 0],
        detected_trees[:, 1],
        color="red",
        s=10,
        label="Detected Trees",
    )
    ax.set_title(f"Point Cloud with Detected Trees - {plot_name}")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{plot_name}_point_cloud_with_trees.png")
    plt.savefig(plot_path, bbox_inches="tight")
    logger.info(f"Point cloud with detected trees plot saved to {plot_path}")
    plt.close()


def process_all_las_files_with_ground_truth(
    las_dir,
    ground_truth_data,
    save_dir,
    max_distance=5.0,
    max_height_difference=3.0,
    window_size=2.0,
):
    plots_dir = os.path.join(save_dir, "plots")
    geojson_dir = os.path.join(save_dir, "geojson")
    point_cloud_with_trees_dir = os.path.join(save_dir, "point_cloud_with_trees")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(geojson_dir, exist_ok=True)
    os.makedirs(point_cloud_with_trees_dir, exist_ok=True)

    plot_dirs = sorted(os.listdir(las_dir))
    metrics_list = []

    for plot_dir in plot_dirs:
        plot_name = os.path.basename(plot_dir)

        ground_points = np.load(os.path.join(las_dir, plot_dir, f"{plot_name}_ground.npy"))
        vegetation_points = np.load(os.path.join(las_dir, plot_dir, f"{plot_name}_vegetation.npy"))

        logger.info(f"Processing {plot_name}...")
        if vegetation_points.size == 0:
            logger.warning(f"No vegetation points found for {plot_name}. Skipping...")
            continue

        detected_trees = detect_trees(vegetation_points, ground_points, window_size)

        plot_ground_truth_np = get_plot_ground_truth(ground_truth_data, int(plot_name.split("_")[1]))
        if plot_ground_truth_np.size == 0:
            logger.warning(f"No ground truth for {plot_name}. Skipping...")
            continue

        detected_trees = crop_by_other(detected_trees, plot_ground_truth_np)
        logger.info(f"Detected trees after cropping: {len(detected_trees)}")

        save_detected_trees_as_geojson(detected_trees, plot_name, geojson_dir)

        plot_point_cloud_with_detected_trees(
            vegetation_points, detected_trees, plot_name, point_cloud_with_trees_dir
        )

        matches = match_candidates(
            ground_truth=plot_ground_truth_np,
            candidates=detected_trees,
            max_distance=max_distance,
            max_height_difference=max_height_difference,
        )

        metrics = calculate_detection_metrics(matches)
        metrics["plot"] = plot_name
        metrics_list.append(metrics)

        plot_path = os.path.join(plots_dir, f"{plot_name}_detection_results.png")
        visualize_detection_results(detected_trees, plot_ground_truth_np, matches, save_path=plot_path)

    metrics_summary = pd.DataFrame(metrics_list)
    metrics_summary.to_csv(os.path.join(save_dir, "detection_metrics.csv"), index=False)
    logger.info("Detection metrics summary saved.")

    return metrics_summary
