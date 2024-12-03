# Parkinson's Insight Engine (PIE)

## Overview
Parkinson's Insight Engine (PIE) is a Python-based data preprocessing and analysis pipeline designed for researchers working with the Michael J. Fox Foundation's Parkinson's Progression Markers Initiative (PPMI) dataset and other MJFF data. PIE enables efficient data cleaning, feature engineering, and preparation for machine learning tasks using multi-modal data (clinical, imaging, biologic, genetic) while ensuring ease of use and reproducibility.

## Features
- **Data Loading**: Loads the multi-modal data and unifies it based on Patient Number (`PATNO`)
- **Data Cleaning**: Handles missing values and outliers and standardizes formats.
- **Feature Engineering**: Aggregates longitudinal data and integrates multi-modal features.
- **Feature Selection**: Automates feature selection.
- **Visualization Tools**: Generates insights through interactive plots and dashboards.
- **Reusable**: Designed as an installable Python package for seamless integration into research workflows.

## Getting Started

### Prerequisites
To use PIE, please ensure you have installed Python 3.8 or later. The pipeline relies on the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `plotly`
- `pytest` (for testing)
  

You can install the required dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/parkinsons-insight-engine.git
cd parkinsons-insight-engine
```

Install the package:

```bash
pip install .
```

### Usage

#### 1. Load Data
Load your MJFF datasets (e.g., CSV files) into PIE:

```python
from pie import DataLoader

data = DataLoader.load("path/to/your/data/folder", source="PPMI")
```

#### 2. Preprocess Data
Clean and standardize the data:

```python
from pie import DataPreprocessor

cleaned_data = DataPreprocessor.clean(data)
```

#### 3. Feature Engineering
Generate new features and integrate across modalities:

```python
from pie import FeatureEngineer

engineered_data = FeatureEngineer.create_features(cleaned_data)
```

#### 4. Feature Selection
Select the features that most discriminate the data:

```python
from pie import FeatureSelector

selected_features = FeatureSelector.select_features(engineered_data, target_column="COHORT")

```

#### 5. Visualization
Generate visualizations:

```python
from pie import Visualizer

Visualizer.plot_distribution(selected_features, column="age")
```

### Example Notebook
Explore the example Jupyter notebook (`examples/pipeline_demo.ipynb`) to see PIE in action.

## Repository Structure
```plaintext
parkinsons-insight-engine/
├── pie/                  # Core library
│   ├── data_loader.py    # Module for loading data
│   ├── data_preprocessor.py # Module for cleaning data
│   ├── feature_engineer.py  # Module for feature engineering
│   ├── feature_selector.py     # Module for feature selection
│   ├── visualizer.py        # Module for data visualization
├── examples/            # Example Jupyter notebooks
├── tests/               # Unit tests
├── requirements.txt     # List of dependencies
├── setup.py             # Installation script
└── README.md            # Project documentation
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Create a pull request.

### Running Tests
Ensure all tests pass before submitting a pull request:

```bash
pytest tests/
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
If you have any questions or suggestions, please don't hesitate to contact cameron@allianceai.co.
