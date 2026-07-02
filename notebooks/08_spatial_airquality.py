import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/cleaned_weather.csv')

# ============================================================
# PART 1: SPATIAL / GEOGRAPHICAL PATTERNS
# ============================================================

# --- Avg temperature by country (top 15 and bottom 15) ---
country_temp = df.groupby('country')['temperature_celsius'].mean().sort_values(ascending=False)
print("=== TOP 10 HOTTEST COUNTRIES (avg) ===")
print(country_temp.head(10))
print("\n=== TOP 10 COLDEST COUNTRIES (avg) ===")
print(country_temp.tail(10))

# --- Scatter: lat/long colored by temperature (simple world map proxy) ---
plt.figure(figsize=(14, 7))
loc_avg = df.groupby(['location_name', 'country', 'latitude', 'longitude'])['temperature_celsius'].mean().reset_index()
sc = plt.scatter(loc_avg['longitude'], loc_avg['latitude'], c=loc_avg['temperature_celsius'],
                  cmap='coolwarm', s=40, edgecolor='k', linewidth=0.3)
plt.colorbar(sc, label='Avg Temperature (°C)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Global Spatial Distribution of Average Temperature by Location')
plt.tight_layout()
plt.savefig('report/figures/11_spatial_temperature_map.png', dpi=150)
plt.close()

# --- Temperature by latitude band (confirms equator-to-pole gradient) ---
loc_avg['lat_band'] = pd.cut(loc_avg['latitude'], bins=[-90, -60, -30, 0, 30, 60, 90],
                               labels=['-90 to -60', '-60 to -30', '-30 to 0', '0 to 30', '30 to 60', '60 to 90'])
lat_band_temp = loc_avg.groupby('lat_band', observed=True)['temperature_celsius'].mean()
print("\n=== AVG TEMPERATURE BY LATITUDE BAND ===")
print(lat_band_temp)

plt.figure(figsize=(9, 5))
lat_band_temp.plot(kind='bar', color='coral')
plt.title('Average Temperature by Latitude Band')
plt.ylabel('Avg Temperature (°C)')
plt.xlabel('Latitude Band')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('report/figures/12_temp_by_latitude_band.png', dpi=150)
plt.close()

# ============================================================
# PART 2: AIR QUALITY vs WEATHER CORRELATION
# ============================================================

aq_cols = [c for c in df.columns if c.startswith('air_quality')]
print(f"\n=== AIR QUALITY COLUMNS FOUND ===\n{aq_cols}")

weather_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'uv_index']
corr_matrix = df[weather_cols + aq_cols].corr()

print("\n=== AIR QUALITY CORRELATIONS WITH WEATHER ===")
print(corr_matrix.loc[aq_cols, weather_cols])

plt.figure(figsize=(11, 7))
sns.heatmap(corr_matrix.loc[aq_cols, weather_cols], annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Air Quality vs Weather Parameter Correlations')
plt.tight_layout()
plt.savefig('report/figures/13_airquality_weather_correlation.png', dpi=150)
plt.close()

# --- Wind speed vs PM2.5 specifically (known physical relationship: wind disperses pollution) ---
if 'air_quality_PM2.5' in df.columns:
    plt.figure(figsize=(9, 6))
    sample = df.sample(min(5000, len(df)), random_state=42)  # sample for readable scatter
    plt.scatter(sample['wind_kph'], sample['air_quality_PM2.5'], alpha=0.3, s=10)
    plt.xlabel('Wind Speed (kph)')
    plt.ylabel('PM2.5')
    plt.title('Wind Speed vs PM2.5 (sampled, n=5000)')
    plt.ylim(0, sample['air_quality_PM2.5'].quantile(0.99))  # trim extreme outliers for readability
    plt.tight_layout()
    plt.savefig('report/figures/14_wind_vs_pm25.png', dpi=150)
    plt.close()

print("\nSaved spatial + air quality figures.")