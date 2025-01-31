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
├── tests/ 
│   └── test_lmf_utils.py  # testing some of the main lmf functions
│
├── main.py                 # Main script to run the entire pipeline
│
├── README.md               # Project README
└── requirements.txt        # Python dependencies
```



### **✅ Updated README Instructions**
```markdown
## Instructions

### **1. Clone the Repository**
First, download the repository to your local machine:

```bash
git clone https://github.com/minanaseh/reproducible-analytical-pipeline.git
cd reproducible-analytical-pipeline
```

---

## **Running the Pipeline with Python**

If you want to run the pipeline in a **Python environment**, follow these steps.

### **2. Install Dependencies**
Set up your Python environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Run the Entire Pipeline (Python)**
Execute the pipeline by running:

```bash
python main.py
```

This will:
- Preprocess ALS data and save results in `results/pre_visualization/`.
- Run Local Maxima Filtering (LMF) and save results in `results/lmf/`.

---

## **Running the Pipeline with Docker**
If you prefer to use **Docker** (recommended for reproducibility), follow these steps.

### **2. Pull the Latest Docker Image (Recommended)**
If you want to use the latest version, you can pull it directly from **Docker Hub**:

```bash
docker pull minanaseh/reproducible-analytical-pipeline:latest
```

### **3. Run the Pipeline in a Docker Container**
```bash
docker run -v $(pwd)/results:/app/results minanaseh/reproducible-analytical-pipeline:latest
```

This will:
- Process ALS data and generate results inside `results/pre_visualization/`.
- Run Local Maxima Filtering (LMF) and save results inside `results/lmf/`.

### **(Optional) Build the Docker Image Locally**
If you have modified the code and want to **build the image locally**, run:

```bash
docker build -t my_pipeline .
```

Then, run it using:
```bash
docker run -v $(pwd)/results:/app/results my_pipeline
```

---





This ensures that all output files are saved to `results/`.

