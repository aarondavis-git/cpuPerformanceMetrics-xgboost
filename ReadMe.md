# CPU Performance Metrics Analysis (XGBoost)

## Overview
This project analyzes a synthetic dataset of CPU and system performance metrics and attempts to predict key system variables using machine learning (XGBoost).

The workflow focuses on:
- Data cleaning and preprocessing
- Outlier detection and handling
- Missing value analysis
- Model training and evaluation
- Interpretation of weak predictive performance

---

## Dataset

The dataset contains 10,000 rows and 7 features:

- Disk Write Speed (MB/s)
- Disk Read Speed (MB/s)
- CPU Usage (%)
- CPU Temperature (°C)
- Clock Speed (GHz)
- Cache Miss Rate (%)
- Power Consumption (W)

Some columns include:
- Missing values
- Corrupted entries
- Outliers and extreme values
- Inconsistent formatting

---

## Data Cleaning Steps

The following preprocessing steps were applied:

### 1. Missing Value Analysis
- Computed percentage of missing values per column
- Overall missing rate ≈ 2%

### 2. Outlier Handling
- CPU Usage constrained to valid range (0–100%)
- Negative values removed or set to NaN
- Extreme values clipped using 99th percentile thresholds

### 3. Data Type Fixes
- Converted object-type columns to numeric where needed

### 4. Final Clean Dataset
- Remaining missing values handled implicitly via XGBoost

---

## Exploratory Data Analysis

### Correlation Analysis
A correlation heatmap was generated to examine relationships between variables.

**Key Finding:**
- All correlations are weak (~0.16–0.17 range)
- No strong linear relationships between features and targets

---

## Model

### Algorithm
- XGBoost Regressor

### Train/Test Split
- 80% training
- 20% testing
- Random state: 42

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## Results

### Example Results (CPU Usage)
- MAE: ~15–22
- RMSE: ~19–49
- Train R²: ~0.04–0.05
- Test R²: ~-0.00 to -0.01

### Power Consumption Results
- Similar performance pattern
- Low predictive power

---

## Feature Importance

Feature importance analysis showed:
- No dominant feature
- Relatively uniform contribution across variables
- Weak signal distribution across all inputs

---

## Key Insight

Despite using a powerful model (XGBoost), performance remains low due to:

- Weak relationships between features and targets
- Likely independent or randomly generated variables
- Limited predictive structure in the dataset

### Conclusion:
> The dataset is better suited for demonstrating data cleaning, preprocessing, and model evaluation techniques than for high-accuracy prediction tasks.

---

## Skills Demonstrated

- Data cleaning and preprocessing (pandas, numpy)
- Outlier detection and handling
- Missing data analysis
- Train/test splitting
- Machine learning with XGBoost
- Model evaluation (MAE, RMSE, R²)
- Data visualization (correlation heatmap, feature importance)

---

## Possible Improvements

- Feature engineering (ratios, interactions)
- Time-series modeling (if temporal structure exists)
- Alternative targets (e.g., CPU Temperature or Usage)
- Comparison with simpler models (linear regression baseline)

---

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib