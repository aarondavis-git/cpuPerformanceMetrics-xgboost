# CPU Performance Metrics Analysis with XGBoost
## Overview
This project analyzes a synthetic dataset of CPU and disk performance metrics using machine learning.
The goal was to predict different target variables from available system metrics and evaluate how well the features explain system behavior.
The project focuses on:
- Data cleaning and preprocessing
- Missing value analysis
- Corrupted value handling
- Outlier detection and treatment
- Exploratory data analysis
- XGBoost regression modeling
- Interpretation of weak model performance
---
## Dataset
The dataset contains 10,000 rows and 7 columns:
- Disk Write Speed (MB/s)
- Disk Read Speed (MB/s)
- CPU Usage (%)
- CPU Temperature (°C)
- Clock Speed (GHz)
- Cache Miss Rate (%)
- Power Consumption (W)
The dataset intentionally contains several data quality problems:
- Missing values
- Corrupted string values
- Inconsistent column types
- Extreme outliers
- Invalid values outside physical limits
---
## Initial Inspection
The project began by inspecting:
- Dataset shape
- Column types
- Duplicate rows
- Missing values
- Sample values from object columns
Two columns were identified as incorrectly typed:
- `Clock Speed (GHz)`
- `Cache Miss Rate (%)`
These columns contained invalid string values such as `"ERROR"` and were converted to numeric using:
```python
pd.to_numeric(..., errors="coerce")

Invalid entries were automatically converted to NaN.

⸻

Missing Value Analysis

Missing values were analyzed both by count and percentage.

Missing Values (%) Per Column

* CPU Usage (%) → ~3.0%
* Disk Write Speed (MB/s) → ~3.0%
* Cache Miss Rate (%) → ~2.5%
* Disk Read Speed (MB/s) → ~2.0%
* CPU Temperature (°C) → ~2.0%
* Clock Speed (GHz) → ~1.5%
* Power Consumption (W) → 0.0%

Overall missing percentage across the dataset was approximately 2%.

⸻

Data Cleaning

Several cleaning rules were applied based on realistic hardware constraints.

CPU Usage (%)

* Restricted to range 0–100%

Clock Speed (GHz)

* Restricted to range 0–10 GHz

Cache Miss Rate (%)

* Restricted to range 0–100%

Disk Speeds

* Negative values removed
* Extreme values clipped at the 99th percentile

Power Consumption (W)

* Negative values removed
* Extreme values clipped at the 99th percentile

After cleaning, approximately 13.6% of rows contained at least one missing value.

Missing values in feature columns were left for XGBoost to handle automatically.

⸻

Exploratory Data Analysis

Correlation Analysis

A correlation heatmap was generated to examine relationships between variables.

Key Finding

All correlations were weak, generally in the range of 0.16–0.17.

This suggests that the features have only weak relationships with each other and with the target variables.

Example correlations for CPU Usage (%):

* Power Consumption (W) → ~0.17
* Clock Speed (GHz) → ~0.17
* Disk Write Speed (MB/s) → ~0.17
* CPU Temperature (°C) → ~0.17
* Cache Miss Rate (%) → ~0.16
* Disk Read Speed (MB/s) → ~0.16

⸻

Machine Learning Model

Algorithm

* XGBoost Regressor

Train/Test Split

* 80% training
* 20% testing
* Random state = 42

Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

⸻

Targets Tested

Three separate targets were tested:

1. Power Consumption (W)
2. CPU Usage (%)
3. CPU Temperature (°C)

⸻

Results

Power Consumption (W)

* MAE ≈ 21.8
* RMSE ≈ 48.7
* Train R² ≈ 0.05
* Test R² ≈ -0.00

CPU Usage (%)

* MAE ≈ 16.0
* RMSE ≈ 19.9
* Train R² ≈ 0.05
* Test R² ≈ -0.01

CPU Temperature (°C)

* MAE ≈ 8.4
* RMSE ≈ 11.1
* Train R² ≈ 0.05
* Test R² ≈ -0.01

⸻

Feature Importance

Feature importance plots were generated for the XGBoost models.

Key Finding

* No feature was strongly dominant
* Importance values were relatively uniform
* This supports the weak correlation results

⸻

Key Insight

Despite using a strong machine learning model, predictive performance remained very low across all targets.

This suggests that:

* The features contain very little predictive signal
* Relationships between variables are weak
* The dataset may have been generated with mostly independent variables
* Better performance is unlikely without stronger features or additional data

Conclusion

This dataset is more useful for demonstrating data cleaning, preprocessing, feature inspection, and machine learning workflow than for building a high-performing predictive model.

⸻

Skills Demonstrated

* Data cleaning with pandas
* Handling corrupted values
* Missing value analysis
* Outlier treatment
* Feature type conversion
* Exploratory data analysis
* Correlation analysis
* XGBoost regression
* Model evaluation with MAE, RMSE, and R²
* Feature importance visualization
* Interpretation of weak predictive performance

⸻

Possible Improvements

* Feature engineering
* Interaction variables
* Ratio-based features
* Simpler baseline models
* Additional visualization
* Residual analysis
* Cross-validation
* Hyperparameter tuning
* Time-series modeling if timestamps become available

⸻

Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
