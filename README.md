# Parkinson's Insight Engine (PIE)

## Overview
Parkinson's Insight Engine (PIE) is a Python-based data preprocessing and analysis pipeline designed for researchers working with the Michael J. Fox Foundation's Parkinson's Progression Markers Initiative (PPMI) dataset and other MJFF data. PIE enables efficient data cleaning, feature engineering, and preparation for machine learning tasks using multi-modal data (clinical, imaging, biologic, genetic) while ensuring ease of use and reproducibility.

## Features
- **Data Loading**: Loads the multi-modal data and unifies it based on Patient Number (`PATNO`)
- **Data Cleaning**: Handles missing values and outliers and standardizes formats.
- **Feature Engineering**: Aggregates longitudinal data and integrates multi-modal features.
- **Feature Selection**: Automates feature selection.
- **Classification and Regression**: Perform supervised learning on the processed data.
- **Visualization Tools**: Generates insights through interactive plots and dashboards.
- **Reusable**: Designed as an installable Python package for seamless integration into research workflows.

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/MJFF-ResearchCommunity/PIE.git
cd PIE
```

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

Install the package:

```bash
pip install .
```

### Data Format

You must [apply for access to the PPMI data](https://www.ppmi-info.org/access-data-specimens/download-data) through the PPMI website. After being granted access individual modalities can be downloaded as separate files through [LONI (the Laboratory of Neuro Imaging from the USC)](https://ida.loni.usc.edu/login.jsp).

For development and testing purposes, this repo assumes data has been downloaded from LONI and stored in a directory called `PPMI`.

When using this package as part of your own development, the path to your local copy of the data can be specified (see Usage section below).

All data are formatted as Pandas DataFrames unless otherwise specified.

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
Explore the example Jupyter notebooks (eg `notebooks/pipeline_demo.ipynb`) to see PIE in action.

## Repository Structure
```plaintext
PIE/
├── pie/                  # Core library
│   ├── __init__.py       # Allow module import
│   ├── data_loader.py    # Module for loading data
│   ├── data_preprocessor.py # Module for cleaning data
│   ├── feature_engineer.py  # Module for feature engineering
│   ├── feature_selector.py  # Module for feature selection
│   ├── visualizer.py        # Module for data visualization
├── notebooks/           # Example Jupyter notebooks
├── tests/               # Unit tests
├── PPMI/                # Local copies of the PPMI data files
├── requirements.txt     # List of dependencies
├── setup.py             # Installation script
└── README.md            # Project documentation
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Copy the PPMI data into the `PIE/PPMI` directory.
4. Instead of installing PIE as above, use `pip install -e .` for editable mode.
5. Make your changes, and ensure the full test suite runs without failures (see below).
6. Commit your changes: `git commit -m 'Add new feature'`.
7. Push to the branch: `git push origin feature-name`.
8. Create a pull request.

### Running Tests
Ensure all tests pass before submitting a pull request:

```bash
pytest tests/
```

Please also add tests to cover the new code or feature in your pull request. The existing tests can be used as a guideline for what and how to test.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
If you have any questions or suggestions, please don't hesitate to contact cameron@allianceai.co.
