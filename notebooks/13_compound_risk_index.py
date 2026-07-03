"""
Combines z-scored deviations across temp/humidity/wind/pressure into a
single interpretable "compound extreme weather score" — more meaningful
than raw Isolation Forest output, and validated against a real, documented
extreme weather event rather than just eyeballing top anomaly rows.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/cleaned_weather.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

# Z-score each variable within its OWN location — "extreme for Reykjavik" is
# a very different absolute number than "extreme for Dubai". Standardizing
# globally would just re-detect "hot countries are hot", which is not
# interesting. Standardizing per-location surfaces genuine anomalies relative
# to that place's own normal.
features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm']

def zscore_group(g):
    for col in features:
        mu, sigma = g[col].mean(), g[col].std()
        g[f'{col}_z'] = (g[col] - mu) / sigma if sigma > 0 else 0
    return g

df = df.groupby('location_name', group_keys=False).apply(zscore_group, include_groups=True)

# Compound score = combined magnitude of deviation across all 5 signals.
# Using absolute z-scores summed (not squared) so it stays interpretable:
# "this reading is, on average, N standard deviations away from normal
# across these 5 variables at once."
z_cols = [f'{c}_z' for c in features]
df['compound_risk_score'] = df[z_cols].abs().mean(axis=1)

print("=== TOP 15 HIGHEST COMPOUND RISK READINGS ===")
top = df.nlargest(15, 'compound_risk_score')[
    ['location_name', 'country', 'last_updated', 'temperature_celsius',
     'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'compound_risk_score']
]
print(top.to_string(index=False))

# --- Validation against a real, documented event ---
# Cyclone Remal made landfall in Bangladesh/West Bengal on 26-27 May 2024 —
# a real, publicly documented extreme weather event. If our compound risk
# index has genuine signal, Bangladesh readings around this date should show
# elevated compound_risk_score compared to Bangladesh's own baseline.
bd = df[df['country'] == 'Bangladesh'].copy()
if len(bd) > 0:
    bd['date'] = bd['last_updated'].dt.date
    remal_window = bd[(bd['date'] >= pd.Timestamp('2024-05-24').date()) &
                       (bd['date'] <= pd.Timestamp('2024-05-29').date())]
    baseline = bd[~bd.index.isin(remal_window.index)]

    print(f"\n=== VALIDATION: Cyclone Remal (Bangladesh, ~26-27 May 2024) ===")
    print(f"Bangladesh readings in Remal window (24-29 May): {len(remal_window)}")
    print(f"  Mean compound_risk_score during window: {remal_window['compound_risk_score'].mean():.3f}")
    print(f"Bangladesh readings outside window (baseline): {len(baseline)}")
    print(f"  Mean compound_risk_score baseline: {baseline['compound_risk_score'].mean():.3f}")
    if len(remal_window) > 0 and len(baseline) > 0:
        lift = remal_window['compound_risk_score'].mean() / baseline['compound_risk_score'].mean()
        print(f"  Risk score lift during documented cyclone: {lift:.2f}x baseline")
else:
    print("\nNo Bangladesh rows found for validation.")

# Save + visualize distribution
df[['location_name', 'country', 'last_updated', 'compound_risk_score']].to_csv(
    'data/compound_risk_scores.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['compound_risk_score'], bins=60, color='#c0392b', alpha=0.75)
ax.axvline(df['compound_risk_score'].quantile(0.98), color='black', linestyle='--',
           label='98th percentile (extreme threshold)')
ax.set_xlabel('Compound Risk Score (mean |z-score| across 5 variables)')
ax.set_ylabel('Number of readings')
ax.set_title('Distribution of Compound Extreme-Weather Risk Scores')
ax.legend()
plt.tight_layout()
plt.savefig('report/figures/compound_risk_distribution.png', dpi=150)
print("\nSaved data/compound_risk_scores.csv and report/figures/compound_risk_distribution.png")