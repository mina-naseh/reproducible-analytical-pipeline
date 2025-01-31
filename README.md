# Building Reproducible Analytical Pipelines

This repository is a sample part of [this project](https://github.com/mina-naseh/lidar-point-transformer).

This repository implements a pipeline for tree detection and classification using LiDAR data, using Local Maxima Filtering (LMF).

## Dataset

The full dataset is freely available on Kaggle: [Tree Detection Lidar RGB](https://www.kaggle.com/datasets/sentinel3734/tree-detection-lidar-rgb/data). The data includes LiDAR scans and field survey information required for training and evaluation.
For educational reasons of having a simple but reproducible pipeline, we used part of the data for this repository and have pushed it here. The whole data has a big size and is not reasonable to push.
Ideally, the raw data should be handled by the user like [here](https://github.com/mina-naseh/lidar-point-transformer), or using the Kaggle API.

## Repository Structure

```plaintext
REPRODUCIBLE-ANALYTICAL-PIPELINE/
│
├── data/
│   ├── als/                # Raw ALS LiDAR data (downloaded from Kaggle)
│   ├── als_preprocessed/   # Directory created during preprocessing to store processed data
│   └── field_survey.geojson # GeoJSON containing field survey data (ground truth)
│
├── results/                # Consolidated results folder
│   ├── pre_visualization/  # Preprocessing outputs and visualizations
│   ├── lmf/                # Results from Local Maxima Filtering
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── lmf_utils.py         # Utilities for Local Maxima Filtering
│   └── pre_visualization.py # Visualization utilities for preprocessing
│
├── main.py                 # Main script to run the entire pipeline
│
├── README.md               # Project README
└── requirements.txt        # Python dependencies
```

## Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lidar-point-transformer.git
cd lidar-point-transformer
```

### 2. Install Dependencies

Set up your Python environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Entire Pipeline (Python Environment)

Run the `main.py` script to execute all steps of the pipeline sequentially, including preprocessing, Local Maxima Filtering (LMF), and evaluation:

```bash
python main.py
```

This will:

1. Preprocess ALS data and create `results/pre_visualization/`. It also generates visualizations of preprocessed data.
2. Perform Local Maxima Filtering (LMF) and save results in `results/lmf/`.

### 4. Running with Docker

Alternatively, you can run the pipeline using Docker for better reproducibility:

#### Build the Docker Image:
```bash
docker build -t my_pipeline .
```

#### Run the Pipeline in a Docker Container:
```bash
docker run -v $(pwd)/results:/app/results my_pipeline
```

This ensures that all output files are saved to `results/`.

### 5. Results

- **Preprocessing Outputs**: Available in the `results/pre_visualization/` directory, including visualizations of the preprocessed data and statistical summaries.
- **LMF Results**: Available in the `results/lmf/` directory, including detected trees and visualizations.

This setup ensures a fully reproducible analytical pipeline that can be executed in a local Python environment or inside a Docker container.

