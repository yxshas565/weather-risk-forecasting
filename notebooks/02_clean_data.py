import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

df = pd.read_csv('data/GlobalWeatherRepository.csv')

print("=== MISSING VALUES ===")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if len(missing) > 0 else "No missing values found.")

print("\n=== DUPLICATE ROWS ===")
print(f"Duplicates: {df.duplicated().sum()}")

print("\n=== DATE RANGE ===")
df['last_updated'] = pd.to_datetime(df['last_updated'])
print(f"From: {df['last_updated'].min()}  To: {df['last_updated'].max()}")
print(f"Unique dates: {df['last_updated'].dt.date.nunique()}")
print(f"Unique locations: {df['location_name'].nunique()}")
print(f"Unique countries: {df['country'].nunique()}")

print("\n=== NUMERIC SUMMARY (key fields) ===")
key_fields = ['temperature_celsius', 'precip_mm', 'humidity', 'wind_kph', 'pressure_mb', 'uv_index']
print(df[key_fields].describe())

print("\n=== POTENTIAL OUTLIERS (using IQR) ===")
for col in ['temperature_celsius', 'precip_mm', 'wind_kph']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers outside [{lower:.2f}, {upper:.2f}] "
          f"(actual range: {df[col].min():.2f} to {df[col].max():.2f})")