"""
Real forecasting: per-location, multi-step-ahead (next 3 days), instead of
a single smoothed global average. Uses per-location lag features + latitude
(climate baseline) + day-of-year (seasonality).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/cleaned_weather.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])
df = df.sort_values(['location_name', 'last_updated'])

# Keep locations with enough history for meaningful lag/multi-step features
counts = df.groupby('location_name').size()
valid_locations = counts[counts >= 60].index
df = df[df['location_name'].isin(valid_locations)].copy()
print(f"Locations with enough history: {len(valid_locations)} (of {counts.shape[0]} total)")

# Resample to one row per location per day (mean if multiple readings/day)
daily = (
    df.groupby(['location_name', 'country', 'latitude', 'longitude',
                pd.Grouper(key='last_updated', freq='D')])['temperature_celsius']
    .mean()
    .reset_index()
)
daily = daily.dropna(subset=['temperature_celsius'])

# Build lag + calendar + climate-baseline features per location
def build_features(g):
    g = g.sort_values('last_updated').copy()
    for lag in [1, 2, 3, 7]:
        g[f'lag_{lag}'] = g['temperature_celsius'].shift(lag)
    g['rolling_mean_7'] = g['temperature_celsius'].shift(1).rolling(7).mean()
    g['day_of_year'] = g['last_updated'].dt.dayofyear
    # multi-step targets: next 1, 2, 3 days ahead
    g['target_t1'] = g['temperature_celsius'].shift(-1)
    g['target_t2'] = g['temperature_celsius'].shift(-2)
    g['target_t3'] = g['temperature_celsius'].shift(-3)
    return g

daily = daily.groupby('location_name', group_keys=False).apply(build_features)
daily = daily.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'lag_7',
                               'rolling_mean_7', 'target_t1', 'target_t2', 'target_t3'])

print(f"Feature rows after lag/target construction: {len(daily)}")

feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean_7',
                 'day_of_year', 'latitude']

# Time-based split: last 15% of each location's timeline is test data
# (prevents leakage — no shuffling a time series)
daily = daily.sort_values('last_updated')
split_idx = int(len(daily) * 0.85)
train, test = daily.iloc[:split_idx], daily.iloc[split_idx:]
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

results = {}
for horizon, target in [('t+1 day', 'target_t1'), ('t+2 days', 'target_t2'), ('t+3 days', 'target_t3')]:
    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # naive baseline for THIS horizon: lag_1 repeated forward
    naive_preds = X_test['lag_1']
    naive_mae = mean_absolute_error(y_test, naive_preds)

    results[horizon] = {'model_mae': mae, 'model_rmse': rmse, 'model_r2': r2, 'naive_mae': naive_mae}
    beat = "BEATS naive" if mae < naive_mae else "loses to naive"
    print(f"\n{horizon}: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} "
          f"| naive_MAE={naive_mae:.3f} -> model {beat}")

results_df = pd.DataFrame(results).T
results_df.to_csv('report/location_forecast_results.csv')
daily.to_csv('data/location_features.csv', index=False)
print("\nSaved report/location_forecast_results.csv and data/location_features.csv")