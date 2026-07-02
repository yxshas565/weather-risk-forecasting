import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.max_columns', 100)
sns.set_style('whitegrid')

df = pd.read_csv('data/cleaned_weather.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

os.makedirs('report/figures', exist_ok=True)

# --- 1. Temperature distribution ---
plt.figure(figsize=(10, 5))
sns.histplot(df['temperature_celsius'], bins=50, kde=True)
plt.title('Temperature Distribution (Global)')
plt.xlabel('Temperature (°C)')
plt.tight_layout()
plt.savefig('report/figures/01_temp_distribution.png', dpi=150)
plt.close()

# --- 2. Precipitation distribution (log scale since zero-inflated) ---
plt.figure(figsize=(10, 5))
nonzero_precip = df[df['precip_mm'] > 0]['precip_mm']
sns.histplot(nonzero_precip, bins=50, kde=True)
plt.title(f'Precipitation Distribution (nonzero only, n={len(nonzero_precip)})')
plt.xlabel('Precipitation (mm)')
plt.tight_layout()
plt.savefig('report/figures/02_precip_distribution.png', dpi=150)
plt.close()

# --- 3. Temperature trend over time (daily global average) ---
daily_avg = df.groupby(df['last_updated'].dt.date)['temperature_celsius'].mean()
plt.figure(figsize=(14, 5))
daily_avg.plot()
plt.title('Global Average Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Avg Temperature (°C)')
plt.tight_layout()
plt.savefig('report/figures/03_temp_trend.png', dpi=150)
plt.close()

# --- 4. Correlation heatmap (key numeric features) ---
key_numeric = ['temperature_celsius', 'precip_mm', 'humidity', 'wind_kph',
               'pressure_mb', 'uv_index', 'cloud', 'visibility_km',
               'air_quality_PM2.5', 'air_quality_PM10', 'air_quality_us-epa-index']
available_cols = [c for c in key_numeric if c in df.columns]
print(f"Using columns for correlation: {available_cols}")

plt.figure(figsize=(12, 9))
corr = df[available_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix — Key Weather & Air Quality Features')
plt.tight_layout()
plt.savefig('report/figures/04_correlation_heatmap.png', dpi=150)
plt.close()

# --- 5. Temperature by top 10 hottest countries (boxplot) ---
top_hot = df.groupby('country')['temperature_celsius'].mean().sort_values(ascending=False).head(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df['country'].isin(top_hot)], x='country', y='temperature_celsius', order=top_hot)
plt.title('Temperature Spread — Top 10 Hottest Countries (by avg)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('report/figures/05_temp_by_country.png', dpi=150)
plt.close()

print("\nAll 5 figures saved to report/figures/")
print("\n=== TOP CORRELATIONS WITH TEMPERATURE ===")
print(corr['temperature_celsius'].sort_values(ascending=False))