# Weather Trend Forecasting — Data Science Report

**Submitted by:** Yashas Sadananda
**PM Accelerator AI Engineer + Data Science Technical Assessment**

---

## About PM Accelerator

The Product Manager Accelerator Program is designed to support PM professionals
through every stage of their careers — from students becoming product managers
to Directors becoming Chief Product Officers. Our Product Manager Accelerator
community are ambitious and committed, with one thing in common: the desire to
succeed.

---

## 1. Dataset Overview

- **Source:** Kaggle "Global Weather Repository"
- **Rows:** 150,853 daily weather records
- **Locations:** 268 cities across 211 countries
- **Date range:** 2024-05-16 01:45:00 to 2026-07-02 18:45:00
- **Features:** 41 columns covering temperature, precipitation, wind, pressure,
  humidity, UV index, air quality (8 pollutant/index fields), and astronomical
  data (sunrise/sunset/moon phase)

## 2. Data Cleaning & Preprocessing

- **Missing values:** none found
- **Duplicates:** none found
- **Outlier handling:** Used domain knowledge rather than blind IQR filtering,
  since IQR flagged 30k+ rows on `precip_mm` purely because ~67% of days have
  0mm rainfall (a real, expected pattern — not an error). Instead, we removed
  only physically impossible readings:
  - 1 row with temperature_celsius > 60°C (exceeds the highest reliably
    recorded air temperature on Earth, ~54.4°C)
  - 1 row with wind_kph > 300 (a logged wind speed of 2963.2 km/h at
    Bujumbura — faster than a fighter jet, clearly a sensor/logging error)
  - Net: 2 rows dropped out of 150,855 (0.001%)
- Legitimate weather extremes (e.g. genuinely hot desert days) were
  deliberately kept, since removing them would bias later models toward
  artificially calm weather and undermine the anomaly detection step.

## 3. Exploratory Data Analysis

See `report/figures/` for full visualizations (temperature distribution,
precipitation patterns, correlation heatmap, etc.)

**Correlations with temperature:**
| Feature | Correlation |
|---|---|
| UV index | +0.49 |
| Humidity | -0.34 |
| Pressure | -0.29 |
| Wind speed | +0.13 |
| PM10 (air quality) | +0.11 |

Humidity and pressure show the strongest (negative) relationships with
temperature — consistent with basic meteorology (warmer air holds more
moisture but often at lower relative humidity; low-pressure systems often
bring cooler, wetter conditions).

## 4. Forecasting Model

Built a daily global-average-temperature time series (25041 unique days)
using lag features (previous 1/2/3/7 days), rolling statistics, and
calendar features (day of year, month).

**Model comparison (test set, last 154 days held out):**
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.286 | 0.355 | 0.982 |
| Random Forest | 0.317 | 0.404 | 0.977 |
| Gradient Boosting | 0.291 | 0.382 | 0.980 |
| **Ensemble (avg of 3)** | **0.283** | **0.362** | **0.982** |
| Naive baseline (yesterday=today) | 0.263 | 0.337 | 0.984 |

**Honest finding:** the naive baseline (simply predicting tomorrow's global
average temperature equals today's) is competitive with — and slightly
outperforms — all trained models. This is a real and common finding in daily
temperature forecasting: global daily average temperature is highly
autocorrelated day-to-day, so there is limited genuine signal for a model to
extract beyond very recent history. Rather than overstate model performance,
we report this transparently: the value of the modeling exercise here is in
the feature importance and anomaly detection layers (below), not in beating
a strong naive baseline by a wide margin.

## 5. Advanced Analysis

### Anomaly Detection
Using Isolation Forest across temperature, precipitation, humidity, wind,
and pressure: **3,018 anomalous readings identified (2.00% of data)**.

Top anomalies cluster into two patterns:
- **High heat + high humidity + high precipitation** combinations (Dhaka,
  Hanoi) — consistent with monsoon-season extreme weather
- **Cold + very high humidity (near 100%) + moderate wind** combinations
  (Dublin, Reykjavik, Vestmannaeyjar) — consistent with North Atlantic
  winter storm systems

### Feature Importance (SHAP)
| Feature | Mean \|SHAP value\| |
|---|---|
| Yesterday's temperature (lag_1) | 1.65 |
| 7-day rolling mean | 1.05 |
| 2 days ago (lag_2) | 0.33 |
| 3 days ago (lag_3) | 0.14 |
| Day of year | 0.04 |

Confirms the autocorrelation finding above: recent temperature history
dominates predictive power far more than calendar/seasonal features.

### Spatial & Geographic Patterns
**Hottest countries (avg temp):** Gulf/desert nations (Qatar, UAE, Oman,
Djibouti) — physically expected.

**Coldest countries (avg temp):** Nordic/high-latitude nations (Iceland,
Canada, Mongolia, Norway) — physically expected.

**By latitude band:**
| Latitude | Avg Temp (°C) |
|---|---|
| 0° to 30° (tropical) | 27.0 |
| -30° to 0° | 23.9 |
| 30° to 60° | 19.8 |
| -60° to -30° | 12.0 |
| 60° to 90° (polar) | 9.0 |

Confirms the expected equator-to-pole temperature gradient.

### Air Quality × Weather Correlation
Ozone shows the strongest relationship with weather conditions: positively
correlated with temperature (+0.25) and UV index (+0.35), negatively with
humidity (-0.40) — consistent with photochemical ozone formation being
temperature/sunlight-driven.

## 6. Key Insights & Limitations

- Daily global temperature is highly persistent; naive forecasting is a
  strong baseline and any production system should benchmark against it
  honestly rather than assume ML complexity implies better performance
- Anomaly detection surfaced physically coherent extreme-weather clusters,
  suggesting the feature set (temp + precip + humidity + wind + pressure
  together) captures real compound-risk conditions, not just univariate
  outliers
- Known data quality caveat: a small number of `country` values appear in
  non-English transliterations (e.g. "Saudi Arabien", "Turkménistan") in
  the raw dataset — worth normalizing before any production use, though it
  did not affect the numeric analyses above

## 7. Files in This Repository
weather-risk-forecasting/
├── data/
│   ├── GlobalWeatherRepository.csv   (raw)
│   └── cleaned_weather.csv           (cleaned)
├── notebooks/
│   ├── 01_explore_data.py
│   ├── 02_clean_data.py
│   ├── 03_clean_and_save.py
│   ├── 04_eda.py
│   ├── 05_forecast_model.py
│   ├── 06_anomaly_detection.py
│   ├── 07_ensemble_shap.py
│   ├── 08_spatial_airquality.py
│   └── 09_generate_report.py
├── report/
│   ├── figures/          (all generated plots)
│   └── report.md         (this file)
└── README.md
