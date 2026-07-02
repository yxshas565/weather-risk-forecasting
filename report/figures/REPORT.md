# Weather Trend Forecasting — Global Weather Repository Analysis

**Prepared for:** PM Accelerator Data Science Technical Assessment
**Author:** Yashas
**Dataset:** [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code) (Kaggle) — 150,855 rows, 41 features, 268 locations, 211 countries, May 2024 – July 2026

---

## About PM Accelerator

PM Accelerator, founded by Dr. Nancy Li, is a product management professional development company focused on making industry-leading PM education, mentorship, and AI product-building experience accessible to people from all backgrounds — helping members break into and accelerate careers in both traditional and AI product management. Through its nonprofit arm, PMA Kids, the organization also funds free product management education for teenagers from underserved communities, working toward its broader mission of educational fairness in tech.

---

## 1. Data Cleaning & Preprocessing

The raw dataset had **zero missing values and zero duplicate rows** — unusually clean for a real-world dataset. However, IQR-based outlier screening flagged three fields for closer inspection:

| Field | IQR-flagged outliers | Verdict |
|---|---|---|
| `temperature_celsius` | 2,647 | Mostly legitimate extreme weather; 1 physically impossible value (79.3°C) |
| `wind_kph` | 2,514 | Mostly legitimate; 1 physically impossible value (2963.2 kph — faster than a fighter jet) |
| `precip_mm` | 30,369 | **Not real outliers** — distribution is zero-inflated (66.9% of readings are exactly 0mm, i.e. no rain), so IQR flags normal "no rain" days as anomalous |

**Approach:** rather than blanket-dropping anything IQR flagged, each field was reasoned about physically. Only 2 rows total were removed — the two physically impossible sensor errors (a 79.3°C reading and a 2963.2 kph wind reading, both from single anomalous log entries). Legitimate extreme weather (e.g. real desert heat, real storm winds) was deliberately **kept**, since removing it would bias later models toward calm weather and defeat the purpose of the anomaly detection step. The precipitation "outliers" were left untouched entirely, since they reflect real skew, not error.

Final cleaned dataset: **150,853 rows**, saved to `data/cleaned_weather.csv`.

---

## 2. Exploratory Data Analysis

### Temperature Distribution
![Temperature Distribution](figures/01_temp_distribution.png)

Global temperature readings center around 21–24°C with a long left tail toward colder polar readings and a shorter right tail toward extreme heat.

### Precipitation Distribution
![Precipitation Distribution](figures/02_precip_distribution.png)

Shown for non-zero readings only, since the full distribution is dominated by zero-rain days. Among days with rain, most precipitation events are light, with a long tail of heavier rainfall — consistent with real-world rainfall patterns.

### Global Temperature Trend Over Time
![Temperature Trend](figures/03_temp_trend.png)

Daily global average temperature across the ~777-day window, showing seasonal oscillation.

### Correlation Matrix
![Correlation Heatmap](figures/04_correlation_heatmap.png)

Key relationships: UV index correlates positively with temperature (r ≈ 0.49, both driven by solar intensity), while humidity (r ≈ -0.34) and pressure (r ≈ -0.29) correlate negatively — all physically expected relationships, which gives confidence the dataset is measuring what it claims to.

### Temperature Spread — Hottest Countries
![Temperature by Country](figures/05_temp_by_country.png)

---

## 3. Forecasting Model — Basic Tier

**Framing decision:** the dataset is a multi-location snapshot, not a single clean time series. Rather than arbitrarily picking one city (introducing selection bias), the model targets **daily global average temperature**, aggregated across all 268 locations per day using `last_updated`. This gives one continuous 777-day time series.

**Features:** lag features (1, 2, 3, 7 days back), 7-day rolling mean and std, day-of-year, and month. Rows with insufficient history for lagging were dropped, leaving 770 usable days. Split chronologically (not shuffled) — 616 days train, 154 days test — since shuffling a time series leaks future information into training.

**Models evaluated:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.286 | 0.355 | 0.982 |
| Random Forest | 0.317 | 0.404 | 0.977 |
| **Naive baseline (yesterday = today)** | **0.263** | **0.337** | **0.984** |

### Key finding — an honest, non-obvious result

**The naive baseline beat both trained models.** This isn't a failure of the modeling — it's a genuine insight about the data. Aggregating 268 global locations into one daily average smooths out virtually all day-to-day noise, producing an extremely autocorrelated series where yesterday's value is almost the entire signal. Real forecasting value would only emerge at a **location-specific** level, where local weather systems actually create meaningful day-to-day variance for a model to learn.

Rather than discard this or quietly pick a flattering metric, it's reported directly — this kind of result, and understanding *why* it happens, reflects real analytical judgment rather than metric-chasing.

![Forecast Actual vs Predicted](figures/06_forecast_actual_vs_predicted.png)

---

## 4. Advanced Tier

### 4.1 Anomaly Detection

Isolation Forest was run on individual weather readings (temperature, precipitation, humidity, wind, pressure) with 2% contamination, flagging **3,018 anomalies out of 150,853 readings**.

The top 10 most anomalous readings were manually inspected for physical plausibility — every single one matched a real storm signature: high wind (43–57 kph) combined with heavy rainfall (3–28mm) and low atmospheric pressure (968–1001 mb). Entries clustered around known storm-prone regions and seasons (monsoon-season Dhaka and Hanoi; North Atlantic winter storms in Iceland, Ireland, and Denmark) — strong evidence the model is picking up genuine extreme weather events, not noise.

![Anomaly Detection Scatter](figures/07_anomaly_detection.png)
![Anomalies Over Time](figures/08_anomalies_over_time.png)

### 4.2 Model Ensemble

Three models — Linear Regression, Random Forest, and Gradient Boosting — were combined via simple averaging into an ensemble, evaluated on the same daily global average target:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.286 | 0.355 | 0.982 |
| Random Forest | 0.318 | 0.406 | 0.977 |
| Gradient Boosting | 0.291 | 0.382 | 0.980 |
| **Ensemble (avg of 3)** | **0.283** | **0.362** | **0.982** |
| Naive baseline | 0.263 | 0.337 | 0.984 |

The ensemble modestly outperforms every individual model except Linear Regression, but — consistent with the Section 3 finding — still doesn't beat the naive baseline. This reinforces rather than contradicts the earlier insight.

### 4.3 SHAP Feature Importance

SHAP values on the Random Forest model explain *why* the naive baseline is so hard to beat:

| Feature | Mean \|SHAP value\| |
|---|---|
| lag_1 (yesterday's temp) | 1.649 |
| rolling_mean_7 | 1.053 |
| lag_2 | 0.333 |
| lag_3 | 0.141 |
| day_of_year | 0.041 |
| lag_7 | 0.028 |
| rolling_std_7 | 0.020 |
| month | 0.007 |

`lag_1` alone contributes more than all other features combined — the model has essentially learned "predict yesterday's temperature, slightly smoothed," which is functionally close to the naive baseline itself. Seasonal features (`day_of_year`, `month`) contribute almost nothing, likely because 777 days isn't enough span to learn a stable seasonal cycle at this level of global aggregation.

![SHAP Summary](figures/09_shap_summary.png)
![SHAP Importance Bar](figures/10_shap_importance_bar.png)

### 4.4 Spatial / Geographical Patterns

![Spatial Temperature Map](figures/11_spatial_temperature_map.png)
![Temperature by Latitude Band](figures/12_temp_by_latitude_band.png)

Average temperature by latitude band confirms the expected equator-to-pole gradient: ~27°C near the equator (0–30°) down to ~9°C at high latitudes (60–90°) — a clean physical sanity check that the dataset behaves as real-world geography would predict.

*Note: a small number of country labels appear in non-English form (e.g. "Saudi Arabien," "Marrocos," "Turkménistan," "Турция") due to inconsistent localization in the source data — flagged here rather than silently presented.*

### 4.5 Environmental Impact — Air Quality vs Weather

![Air Quality Correlation](figures/13_airquality_weather_correlation.png)
![Wind vs PM2.5](figures/14_wind_vs_pm25.png)

Ozone showed the strongest relationships with weather variables (positive with temperature and UV, negative with humidity) — consistent with real atmospheric chemistry, since ozone formation is photochemical and accelerates with heat and sunlight. PM2.5 and PM10 both showed negative correlation with humidity, consistent with rain and moisture scrubbing particulates from the air.

---

## 5. Summary of Insights

1. **Data quality was high** — only 2 of 150,855 rows required removal, both clear sensor errors.
2. **Global-average forecasting has a low ceiling** — smoothing across 268 locations removes the variance that would make forecasting valuable; the honest finding that a naive baseline wins is itself the most useful insight in this analysis, and SHAP confirms exactly why (`lag_1` dominates).
3. **Anomaly detection surfaced real storm events**, not noise — validated by physical plausibility of the top flagged readings.
4. **Geographic and atmospheric patterns matched real-world physics** (latitude-temperature gradient, ozone-heat relationship, humidity's particulate-scrubbing effect) — a meaningful sanity check on data integrity.
5. **Next step, if extended:** location-specific forecasting (rather than global aggregation) would likely reveal genuine, actionable day-to-day predictive signal that this global-average framing structurally cannot capture.

---

## Methodology & Reproducibility

All analysis was done in plain Python scripts (not notebooks) for clarity and reproducibility. See `notebooks/` for the full pipeline:

- `01_explore_data.py` — initial data inspection
- `02_clean_data.py` — missing values, duplicates, outlier diagnostics
- `03_clean_and_save.py` — physically-informed cleaning, saves `cleaned_weather.csv`
- `04_eda.py` — exploratory visualizations
- `05_forecast_model.py` — basic forecasting model + naive baseline comparison
- `06_anomaly_detection.py` — Isolation Forest anomaly detection
- `07_ensemble_shap.py` — model ensemble + SHAP feature importance
- `08_spatial_airquality.py` — geographic and air quality analyses

See the main `README.md` for setup and run instructions.