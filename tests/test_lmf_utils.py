import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point
from src.lmf_utils import (
    local_maxima_filter,
    process_point_cloud_with_lmf,
    transform_ground_truth,
    crop_by_other,
    match_candidates,
    calculate_detection_metrics
)

# Sample point cloud data for testing
POINT_CLOUD = np.array([
    [0, 0, 10],
    [1, 1, 15],
    [2, 2, 20],
    [3, 3, 5],
    [4, 4, 25],
])

GROUND_TRUTH = np.array([
    [0, 0, 10],
    [2, 2, 20],
    [4, 4, 25],
])

CANDIDATES = np.array([
    [0, 0, 10],
    [2, 2, 19],  # Slightly different height
    [5, 5, 30],
])

@pytest.mark.parametrize("window_size, expected_count", [
    (2.0, 3),  # Expect 3 trees detected
    (1.0, 5),  # Expect 5 trees detected (each point is a local max)
])
def test_local_maxima_filter(window_size, expected_count):
    result = local_maxima_filter(POINT_CLOUD, window_size)

    # Debugging: Print the detected trees
    print(f"Window Size: {window_size}, Expected: {expected_count}, Got: {len(result)}")
    print(result)

    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 3  # Ensure output shape is (N, 3) (X, Y, Z)
    assert len(result) == expected_count


def test_local_maxima_filter_empty():
    result = local_maxima_filter(np.array([]), 2.0)
    assert result.size == 0  # Should return an empty array

def test_process_point_cloud_with_lmf():
    result = process_point_cloud_with_lmf(POINT_CLOUD, 2.0)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 3

def test_process_point_cloud_with_lmf_empty():
    result = process_point_cloud_with_lmf(np.array([]), 2.0)
    assert result.size == 0

def test_transform_ground_truth():
    gdf = gpd.GeoDataFrame({
        "geometry": [Point(0, 0), Point(2, 2), Point(4, 4)],
        "height": [10, 20, 25]
    })
    transformed = transform_ground_truth(gdf)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (3, 3)  # Should have 3 columns (X, Y, height)

def test_crop_by_other():
    cropped = crop_by_other(POINT_CLOUD, GROUND_TRUTH)
    assert isinstance(cropped, np.ndarray)
    assert cropped.shape[1] == 3  # Should still be (X, Y, Z)

def test_match_candidates():
    matches = match_candidates(GROUND_TRUTH, CANDIDATES, max_distance=2.0, max_height_difference=2.0)
    assert isinstance(matches, list)
    assert all(isinstance(m, dict) for m in matches)
    assert any(m["class"] == "TP" for m in matches)  # Should contain at least some True Positives

def test_match_candidates_no_candidates():
    matches = match_candidates(GROUND_TRUTH, np.array([]), max_distance=2.0, max_height_difference=2.0)
    assert all(m["class"] == "FN" for m in matches)  # All should be False Negatives

def test_match_candidates_no_ground_truth():
    matches = match_candidates(np.array([]), CANDIDATES, max_distance=2.0, max_height_difference=2.0)
    assert all(m["class"] == "FP" for m in matches)  # All should be False Positives

def test_calculate_detection_metrics():
    matches = [
        {"ground_truth": (0, 0, 10), "candidate": (0, 0, 10), "class": "TP", "distance": 0},
        {"ground_truth": (2, 2, 20), "candidate": None, "class": "FN", "distance": None},
        {"ground_truth": None, "candidate": (5, 5, 30), "class": "FP", "distance": None},
    ]
    metrics = calculate_detection_metrics(matches)
    assert isinstance(metrics, dict)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "mean_distance" in metrics
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
