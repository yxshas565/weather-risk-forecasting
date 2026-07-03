"""
Uncertainty quantification: instead of a single point prediction, fit
quantile regression models at 5% and 95% to produce a genuine prediction
interval ("24.3C, 90% confidence between 22.1-26.5C") rather than a bare
number with no sense of how much to trust it.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

daily = pd.read_csv('data/location_features.csv')
daily['last_updated'] = pd.to_datetime(daily['last_updated'])
daily = daily.sort_values('last_updated')

feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean_7', 'day_of_year', 'latitude']
split_idx = int(len(daily) * 0.85)
train, test = daily.iloc[:split_idx], daily.iloc[split_idx:]

X_train, y_train = train[feature_cols], train['target_t1']
X_test, y_test = test[feature_cols], test['target_t1']

# Point prediction (median-equivalent, same as script 14's main model)
model_point = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
model_point.fit(X_train, y_train)
preds_point = model_point.predict(X_test)

# Lower bound (5th percentile) and upper bound (95th percentile) — this is
# what makes it a genuine 90% prediction interval, not just point estimate
# +/- a fixed number.
model_lower = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42,
                                          loss='quantile', alpha=0.05)
model_lower.fit(X_train, y_train)
preds_lower = model_lower.predict(X_test)

model_upper = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42,
                                          loss='quantile', alpha=0.95)
model_upper.fit(X_train, y_train)
preds_upper = model_upper.predict(X_test)

# Sanity fix: quantile models can occasionally cross (lower > upper) on a
# few rows since they're trained independently — clip to guarantee validity
preds_lower_fixed = np.minimum(preds_lower, preds_point)
preds_upper_fixed = np.maximum(preds_upper, preds_point)

# --- Calibration check: does the 90% interval actually contain the true
# value ~90% of the time? This is the real test of whether the uncertainty
# estimate is honest, not just decorative. ---
within_interval = (y_test >= preds_lower_fixed) & (y_test <= preds_upper_fixed)
coverage = within_interval.mean()
print(f"=== PREDICTION INTERVAL CALIBRATION ===")
print(f"Target coverage: 90%")
print(f"Actual coverage: {coverage*100:.1f}%")
if 0.85 <= coverage <= 0.95:
    print("Well-calibrated — the interval is honest, not just wide for show.")
else:
    print("Miscalibrated — interval width doesn't match its stated confidence level.")

avg_width = (preds_upper_fixed - preds_lower_fixed).mean()
print(f"\nAverage interval width: {avg_width:.2f}°C")

# Save results
results = test[['location_name', 'country', 'last_updated']].copy()
results['actual'] = y_test.values
results['predicted'] = preds_point
results['lower_90'] = preds_lower_fixed
results['upper_90'] = preds_upper_fixed
results['within_interval'] = within_interval.values
results.to_csv('report/prediction_intervals.csv', index=False)
print("\nSaved report/prediction_intervals.csv")

# Visualize for one location as an example
example_loc = results['location_name'].mode()[0]  # most-represented location in test set
ex = results[results['location_name'] == example_loc].sort_values('last_updated').tail(60)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ex['last_updated'], ex['actual'], label='Actual', color='#1a1a2e', linewidth=1.5)
ax.plot(ex['last_updated'], ex['predicted'], label='Predicted', color='#00bbf9', linewidth=1.5)
ax.fill_between(ex['last_updated'], ex['lower_90'], ex['upper_90'],
                 color='#00bbf9', alpha=0.2, label='90% Prediction Interval')
ax.set_title(f'Forecast with Uncertainty — {example_loc} (last 60 test-set days)')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/figures/prediction_intervals_example.png', dpi=150)
print(f"Saved report/figures/prediction_intervals_example.png (example: {example_loc})")