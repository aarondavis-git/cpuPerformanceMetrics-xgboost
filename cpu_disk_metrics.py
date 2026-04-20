import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# -----------------------------
# Load Dataset
# -----------------------------
raw_df = pd.read_csv("unclean_cpu_disk_metrics.csv")


# -----------------------------
# Initial Inspection
# -----------------------------
print("Shape:")
print(raw_df.shape)

print("\nInfo:")
raw_df.info()

print("\nHead:")
print(raw_df.head())

print("\nDuplicate Rows:")
print(raw_df.duplicated().sum())

print("\nClock Speed Sample:")
print(raw_df["Clock Speed (GHz)"].sample(20))

print("\nCache Miss Rate Sample:")
print(raw_df["Cache Miss Rate (%)"].sample(20))


# -----------------------------
# Convert Object Columns to Numeric
# -----------------------------
raw_df["Clock Speed (GHz)"] = pd.to_numeric(
    raw_df["Clock Speed (GHz)"], errors="coerce"
)

raw_df["Cache Miss Rate (%)"] = pd.to_numeric(
    raw_df["Cache Miss Rate (%)"], errors="coerce"
)


# -----------------------------
# Missing Value Analysis
# -----------------------------
print("\nMissing Values Count:")
print(raw_df.isnull().sum())

print("\nUpdated Info:")
raw_df.info()

missing_percent = raw_df.isnull().mean() * 100
missing_summary = missing_percent.sort_values(ascending=False)

print("\nMissing Values (%) Per Column:")
print(missing_summary)

print("\nOverall Missing Percentage:")
print(raw_df.isnull().values.mean() * 100)


# -----------------------------
# Summary Statistics
# -----------------------------
print("\nSummary Statistics:")
print(raw_df.describe())


# -----------------------------
# Data Cleaning
# -----------------------------
df = raw_df.copy()

# CPU Usage (%): valid range 0-100
df.loc[(df["CPU Usage (%)"] < 0) | (df["CPU Usage (%)"] > 100), "CPU Usage (%)"] = (
    np.nan
)

# Disk Speeds: remove negatives and clip upper outliers
for col in ["Disk Write Speed (MB/s)", "Disk Read Speed (MB/s)"]:
    df.loc[df[col] < 0, col] = np.nan
    upper_limit = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper_limit)

# Power Consumption: remove negatives and clip upper outliers
df.loc[df["Power Consumption (W)"] < 0, "Power Consumption (W)"] = np.nan

power_upper_limit = df["Power Consumption (W)"].quantile(0.99)
df["Power Consumption (W)"] = df["Power Consumption (W)"].clip(upper=power_upper_limit)

# Clock Speed: valid range 0-10 GHz
df.loc[
    (df["Clock Speed (GHz)"] < 0) | (df["Clock Speed (GHz)"] > 10), "Clock Speed (GHz)"
] = np.nan

# Cache Miss Rate: valid range 0-100%
df.loc[
    (df["Cache Miss Rate (%)"] < 0) | (df["Cache Miss Rate (%)"] > 100),
    "Cache Miss Rate (%)",
] = np.nan


# -----------------------------
# Post-Cleaning Summary
# -----------------------------
rows_with_nan = df.isna().any(axis=1).sum()
affected_row_percent = df.isna().any(axis=1).mean() * 100

print("\nRows With At Least One NaN:")
print(rows_with_nan)

print("\nPercentage of Affected Rows:")
print(affected_row_percent)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Data Summary:")
print(df.describe())


# -----------------------------
# Model Training Function
# -----------------------------
def run_xgboost_model(dataframe, target_column):
    print("\n" + "=" * 50)
    print(f"Target: {target_column}")
    print("=" * 50)

    df_model = dataframe.dropna(subset=[target_column])

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTrain/Test Shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nMissing Values in X_train:")
    print(X_train.isna().sum())

    print("\nMissing Values in X_test:")
    print(X_test.isna().sum())

    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=1,
        reg_lambda=3,
        random_state=42,
    )

    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    print("\nModel Performance:")
    print("MAE:", mean_absolute_error(y_test, test_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))
    print("Train R2:", r2_score(y_train, train_pred))
    print("Test R2:", r2_score(y_test, test_pred))

    importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values()

    return model, importance


# -----------------------------
# Run Models
# -----------------------------
power_model, power_importance = run_xgboost_model(df, "Power Consumption (W)")

cpu_usage_model, cpu_usage_importance = run_xgboost_model(df, "CPU Usage (%)")

cpu_temp_model, cpu_temp_importance = run_xgboost_model(df, "CPU Temperature (°C)")


# -----------------------------
# Correlation Heatmap
# -----------------------------
corr = df.corr(numeric_only=True)

plt.figure()
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()


# -----------------------------
# Feature Importance Plot
# -----------------------------
plt.figure()
plt.barh(cpu_temp_importance.index, cpu_temp_importance.values)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance")
plt.show()
