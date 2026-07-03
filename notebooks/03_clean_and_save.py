import pandas as pd
import numpy as np

df = pd.read_csv('data/GlobalWeatherRepository.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

print(f"Starting rows: {len(df)}")

# --- Physically impossible values = real data errors, not "extreme weather" ---
# Highest reliably recorded air temp on Earth: ~54.4C (Death Valley). Anything
# above 60C is almost certainly a sensor/logging error, not real weather.
temp_errors = df[(df['temperature_celsius'] > 60) | (df['temperature_celsius'] < -60)]
print(f"Impossible temperature rows: {len(temp_errors)}")

# Highest surface wind gust ever recorded: ~408 km/h (tornado-adjacent, extreme
# rare event). Sustained wind_kph above 300 in a general weather dataset is
# almost certainly a unit/sensor error.
wind_errors = df[df['wind_kph'] > 150]
print(f"Impossible wind_kph rows: {len(wind_errors)}")
print(f"Sample bad wind rows:\n{wind_errors[['location_name','wind_kph','last_updated']].head()}")

# Real-world sea-level-adjusted pressure range is roughly 870-1085 mb
# (highest ever recorded ~1085 mb in a Siberian high, lowest ~870 mb inside
# a typhoon eye). Anything outside this is a sensor/logging error, same
# category as the temp/wind fixes above.
pressure_errors = df[(df['pressure_mb'] > 1085) | (df['pressure_mb'] < 870)]
print(f"Impossible pressure_mb rows: {len(pressure_errors)}")
print(f"Sample bad pressure rows:\n{pressure_errors[['location_name','pressure_mb','last_updated']].head()}")

# Drop only the physically-impossible rows — keep legitimate extremes (e.g. a
# real 48C day) since removing those would bias the model toward calm weather
# and defeats the purpose of anomaly detection later.
df_clean = df[
    (df['temperature_celsius'] <= 60) & (df['temperature_celsius'] >= -60) &
    (df['wind_kph'] <= 150) &
    (df['pressure_mb'] <= 1085) & (df['pressure_mb'] >= 870)
].copy()

print(f"\nRows after removing impossible values: {len(df_clean)} "
      f"(dropped {len(df) - len(df_clean)})")

# --- precip_mm: zero-inflated, not truly "outlier-heavy" ---
# Most locations/days have 0mm rain — that's real, not an error. Leave as-is;
# IQR flagged it because the distribution is skewed, not because it's wrong.
print(f"\nprecip_mm = 0 in {(df_clean['precip_mm']==0).sum()} of {len(df_clean)} rows "
      f"({(df_clean['precip_mm']==0).mean()*100:.1f}%) — expected, not an error.")

# --- Save cleaned dataset ---
df_clean.to_csv('data/cleaned_weather.csv', index=False)
print(f"\nSaved cleaned_weather.csv — {len(df_clean)} rows, {df_clean.shape[1]} columns")