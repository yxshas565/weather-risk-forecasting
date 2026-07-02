import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

daily_model = pd.read_csv('data/daily_temp_features.csv')
daily_model['date'] = pd.to_datetime(daily_model['date'])

feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'day_of_year', 'month']
X = daily_model[feature_cols]
y = daily_model['avg_temp']

split_idx = int(len(daily_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = daily_model['date'].iloc[split_idx:]

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return mae, rmse, r2

# --- Individual models ---
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42).fit(X_train, y_train)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42).fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

# --- Ensemble: simple average of the three ---
ensemble_pred = (lr_pred + rf_pred + gb_pred) / 3

naive_pred = X_test['lag_1'].values

print("=== INDIVIDUAL MODELS ===")
evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)
evaluate("Gradient Boosting", y_test, gb_pred)
print("\n=== ENSEMBLE VS BASELINE ===")
evaluate("Ensemble (avg of 3)", y_test, ensemble_pred)
evaluate("Naive (yesterday=today)", y_test, naive_pred)

# --- SHAP: explain the Random Forest (most interpretable of the non-linear models) ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('report/figures/09_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Also a simple bar chart of mean |SHAP value| per feature
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
print("\n=== MEAN |SHAP VALUE| PER FEATURE ===")
print(mean_abs_shap)

plt.figure(figsize=(8, 5))
mean_abs_shap.plot(kind='barh')
plt.title('Feature Importance (Mean |SHAP value|) — Random Forest')
plt.xlabel('Mean |SHAP value|')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('report/figures/10_shap_importance_bar.png', dpi=150)
plt.close()

print("\nSaved SHAP plots.")