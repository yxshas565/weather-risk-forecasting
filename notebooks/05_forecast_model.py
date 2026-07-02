import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/cleaned_weather.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

# --- Build daily global average time series ---
daily = df.groupby(df['last_updated'].dt.date)['temperature_celsius'].mean().reset_index()
daily.columns = ['date', 'avg_temp']
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values('date').reset_index(drop=True)

print(f"Time series length: {len(daily)} days")
print(daily.head())

# --- Feature engineering: lag features + rolling stats + calendar features ---
for lag in [1, 2, 3, 7]:
    daily[f'lag_{lag}'] = daily['avg_temp'].shift(lag)

daily['rolling_mean_7'] = daily['avg_temp'].shift(1).rolling(window=7).mean()
daily['rolling_std_7'] = daily['avg_temp'].shift(1).rolling(window=7).std()
daily['day_of_year'] = daily['date'].dt.dayofyear
daily['month'] = daily['date'].dt.month

# Drop rows with NaN from lagging/rolling
daily_model = daily.dropna().reset_index(drop=True)
print(f"\nRows after feature engineering (NaN dropped): {len(daily_model)}")

feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'day_of_year', 'month']
X = daily_model[feature_cols]
y = daily_model['avg_temp']

# --- Time-based train/test split (NOT random shuffle — this is a time series) ---
split_idx = int(len(daily_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = daily_model['date'].iloc[split_idx:]

print(f"\nTrain: {len(X_train)} days | Test: {len(X_test)} days")

# --- Model 1: Linear Regression (baseline) ---
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# --- Model 2: Random Forest ---
rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# --- Evaluate ---
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return mae, rmse, r2

print("\n=== MODEL PERFORMANCE (test set) ===")
lr_metrics = evaluate("Linear Regression", y_test, lr_pred)
rf_metrics = evaluate("Random Forest", y_test, rf_pred)

# Naive baseline: "tomorrow = today" for comparison
naive_pred = X_test['lag_1'].values
naive_metrics = evaluate("Naive (yesterday=today)", y_test, naive_pred)

# --- Plot actual vs predicted ---
plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_test.values, label='Actual', linewidth=2)
plt.plot(dates_test, lr_pred, label='Linear Regression', linestyle='--')
plt.plot(dates_test, rf_pred, label='Random Forest', linestyle='--')
plt.title('Global Daily Avg Temperature — Actual vs Predicted (Test Set)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.savefig('report/figures/06_forecast_actual_vs_predicted.png', dpi=150)
plt.close()

daily_model.to_csv('data/daily_temp_features.csv', index=False)
print("\nSaved daily_temp_features.csv and forecast plot.")