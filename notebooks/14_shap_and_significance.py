"""
1. SHAP explainability for the per-location model (the actual best model,
   never explained before now — script 07's SHAP was on the old global model).
2. Statistical significance test: is "model beats naive" real or noise?
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
import shap
import matplotlib.pyplot as plt

daily = pd.read_csv('data/location_features.csv')
daily['last_updated'] = pd.to_datetime(daily['last_updated'])
daily = daily.sort_values('last_updated')

feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean_7', 'day_of_year', 'latitude']
split_idx = int(len(daily) * 0.85)
train, test = daily.iloc[:split_idx], daily.iloc[split_idx:]

# Refit the t+1 model (same config as script 11) so we have a live model
# object to explain — script 11 didn't save the fitted model, only metrics.
model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
X_train, y_train = train[feature_cols], train['target_t1']
X_test, y_test = test[feature_cols], test['target_t1']
model.fit(X_train, y_train)
preds = model.predict(X_test)

# ============================== 1. SHAP ==============================
print("=== SHAP FEATURE IMPORTANCE — per-location t+1 model ===")
explainer = shap.TreeExplainer(model)
# Sample for speed — 2000 rows is plenty for stable SHAP estimates on this feature count
sample = X_test.sample(min(2000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(sample)

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)
print(mean_abs_shap)

fig, ax = plt.subplots(figsize=(9, 5))
mean_abs_shap.plot(kind='barh', ax=ax, color='#00bbf9')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Feature Importance — Per-Location Forecast Model (t+1 day)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('report/figures/location_model_shap.png', dpi=150)
print("Saved report/figures/location_model_shap.png")

# ============================== 2. Significance test ==============================
print("\n=== STATISTICAL SIGNIFICANCE: model vs. naive baseline ===")
naive_preds = X_test['lag_1']

model_errors = np.abs(y_test - preds)
naive_errors = np.abs(y_test - naive_preds)

# Paired test since both models predict the SAME rows — Wilcoxon signed-rank
# is the right choice here (doesn't assume errors are normally distributed,
# which absolute-error distributions usually aren't).
stat, p_value = stats.wilcoxon(model_errors, naive_errors, alternative='less')

print(f"Model mean absolute error: {model_errors.mean():.4f}")
print(f"Naive mean absolute error: {naive_errors.mean():.4f}")
print(f"Wilcoxon signed-rank statistic: {stat:.1f}")
print(f"p-value: {p_value:.2e}")

if p_value < 0.001:
    verdict = "HIGHLY SIGNIFICANT (p < 0.001) — the improvement is real, not noise."
elif p_value < 0.05:
    verdict = "SIGNIFICANT (p < 0.05) — the improvement is real, not noise."
else:
    verdict = "NOT statistically significant — improvement could be due to chance."
print(f"\nVerdict: {verdict}")

# Save results for the report
with open('report/significance_test.txt', 'w') as f:
    f.write(f"Model MAE: {model_errors.mean():.4f}\n")
    f.write(f"Naive MAE: {naive_errors.mean():.4f}\n")
    f.write(f"Wilcoxon statistic: {stat:.1f}\n")
    f.write(f"p-value: {p_value:.2e}\n")
    f.write(f"Verdict: {verdict}\n")
print("Saved report/significance_test.txt")