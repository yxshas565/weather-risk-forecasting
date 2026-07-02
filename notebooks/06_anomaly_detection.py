import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

df = pd.read_csv('data/cleaned_weather.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

# --- Anomaly detection on individual weather readings (not the daily aggregate) ---
# Use a set of features that together define "unusual weather" for a given reading
features = ['temperature_celsius', 'precip_mm', 'humidity', 'wind_kph', 'pressure_mb']
X = df[features].copy()

iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
df['anomaly'] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
df['anomaly_score'] = iso.decision_function(X)  # lower = more anomalous

n_anomalies = (df['anomaly'] == -1).sum()
print(f"Detected {n_anomalies} anomalies out of {len(df)} rows ({n_anomalies/len(df)*100:.2f}%)")

print("\n=== TOP 10 MOST ANOMALOUS READINGS ===")
top_anomalies = df.nsmallest(10, 'anomaly_score')[
    ['location_name', 'country', 'last_updated'] + features + ['anomaly_score']
]
print(top_anomalies.to_string(index=False))

# --- Plot: anomalies overlaid on temperature vs humidity ---
plt.figure(figsize=(10, 7))
normal = df[df['anomaly'] == 1]
anomalies = df[df['anomaly'] == -1]
plt.scatter(normal['temperature_celsius'], normal['humidity'], s=5, alpha=0.3, label='Normal')
plt.scatter(anomalies['temperature_celsius'], anomalies['humidity'], s=20, color='red', label='Anomaly')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title(f'Anomaly Detection (Isolation Forest) — {n_anomalies} anomalies flagged')
plt.legend()
plt.tight_layout()
plt.savefig('report/figures/07_anomaly_detection.png', dpi=150)
plt.close()

# --- Plot: anomalies over time ---
daily_anomaly_count = df[df['anomaly'] == -1].groupby(df['last_updated'].dt.date).size()
plt.figure(figsize=(14, 5))
daily_anomaly_count.plot(kind='bar')
plt.title('Anomalies Detected Per Day')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks([])  # too many days to label individually
plt.tight_layout()
plt.savefig('report/figures/08_anomalies_over_time.png', dpi=150)
plt.close()

df.to_csv('data/weather_with_anomalies.csv', index=False)
print("\nSaved weather_with_anomalies.csv, anomaly plots.")