# Breast Cancer Prediction Project

This project focuses on predicting breast cancer diagnosis using the Wisconsin Breast Cancer Diagnostic dataset. The project includes data analysis, visualization, preprocessing, and predictive modeling.

## Project Structure

```
breast-cancer-prediction/
├── config/                 # Hydra configuration files
├── data/                   # Data storage
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   └── visualization/    # Visualization utilities
└── results/              # Model results and artifacts
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the project:
```bash
python src/main.py
```

## Components

- Data Loading: Fetches the Wisconsin Breast Cancer dataset using UCI ML Repository
- Data Preprocessing: Handles missing values, feature scaling, and feature selection
- Visualization: Creates various plots for data analysis
- Modeling: Implements and evaluates different ML models
- Configuration: Uses Hydra for configuration management

## Features

The dataset includes 30 features related to breast cancer diagnosis:
- radius, texture, perimeter, area
- smoothness, compactness, concavity
- concave points, symmetry, fractal dimension
- (each measured for 3 different regions)

## Target

The target variable is 'Diagnosis' indicating whether the tumor is malignant or benign. 