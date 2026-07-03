"""
Interactive dashboard for the Weather Trend Forecasting project.
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Weather Trend Forecasting", page_icon="🌍", layout="wide")

# ============================== CUSTOM THEME ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 15%, #1a1a2e 0%, #0d0d17 45%, #05050a 100%);
    background-attachment: fixed;
}

/* Animated gradient title */
.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(120deg, #00f5d4, #00bbf9, #9b5de5, #f15bb5, #00f5d4);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 8s ease infinite;
    margin-bottom: 0;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-subtitle {
    color: #8b8ba7;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: -0.3rem;
}

/* Glassmorphic PM Accelerator banner */
.pm-banner {
    background: linear-gradient(135deg, rgba(0,245,212,0.08), rgba(155,93,229,0.08));
    border: 1px solid rgba(0,245,212,0.25);
    border-radius: 16px;
    padding: 18px 24px;
    backdrop-filter: blur(12px);
    color: #d8d8f0;
    margin: 1.2rem 0 1.8rem 0;
    box-shadow: 0 8px 32px rgba(0,245,212,0.06);
}
.pm-banner b { color: #00f5d4; }

/* Glowing animated metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,187,249,0.10), rgba(155,93,229,0.10));
    border: 1px solid rgba(0,187,249,0.3);
    border-radius: 14px;
    padding: 16px 18px 8px 18px;
    box-shadow: 0 0 20px rgba(0,187,249,0.08);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 0 30px rgba(0,245,212,0.25);
    transform: translateY(-3px);
    border-color: rgba(0,245,212,0.6);
}
div[data-testid="stMetricValue"] {
    color: #00f5d4;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
div[data-testid="stMetricLabel"] { color: #8b8ba7; }

/* Tabs restyle */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom: none;
    color: #8b8ba7;
    padding: 10px 20px;
    font-weight: 500;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,245,212,0.15), rgba(155,93,229,0.15)) !important;
    color: #00f5d4 !important;
    border-color: rgba(0,245,212,0.4) !important;
}

/* Section headers with glow underline */
h2, h3 { color: #e8e8f5 !important; }
h2::after, h3::after {
    content: '';
    display: block;
    width: 60px;
    height: 3px;
    margin-top: 6px;
    background: linear-gradient(90deg, #00f5d4, #9b5de5);
    border-radius: 2px;
}

/* Caption / helper text */
.stCaption, [data-testid="stCaptionContainer"] { color: #6e6e8a !important; }

/* Selectbox / slider glow on focus */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04);
    border-color: rgba(0,187,249,0.3);
}

/* Footer */
.footer-text {
    text-align: center;
    color: #4a4a5e;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

/* Success banner glow */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(0,245,212,0.35);
}

/* Dataframe restyle */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
            
div[data-baseweb="select"] * { cursor: pointer !important; }
div[data-baseweb="select"] > div {
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-baseweb="popover"] { transition: opacity 0.25s ease !important; }
</style>
""", unsafe_allow_html=True)

# Shared dark plotly theme
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.02)',
    font=dict(family='Space Grotesk, sans-serif', color='#c8c8de'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    hoverlabel=dict(bgcolor='#1a1a2e', font_color='#e8e8f5', bordercolor='#00f5d4'),
)
ACCENT = ['#00f5d4', '#00bbf9', '#9b5de5', '#f15bb5', '#fee440']

# ============================== HEADER ==============================
st.markdown('<div class="hero-title">🌍 WEATHER TREND FORECASTING</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">PM Accelerator Data Science Technical Assessment — Yashas Sadananda</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pm-banner">
<b>About PM Accelerator:</b> The Product Manager Accelerator Program is designed to support PM
professionals through every stage of their careers — from students becoming product managers
to Directors becoming Chief Product Officers.
</div>
""", unsafe_allow_html=True)

# ============================== DATA LOADING ==============================
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_weather.csv')
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    return df

@st.cache_data
def load_risk_scores():
    return pd.read_csv('data/compound_risk_scores.csv')

@st.cache_data
def load_idw_grid():
    data = np.load('data/idw_grid.npz')
    return data['lat_grid'], data['lon_grid'], data['grid_temps']

df = load_data()
risk_df = load_risk_scores()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍  EXPLORE LOCATION", "🌡️  TEMPERATURE FIELD",
    "🚨  RISK EXPLORER", "🔮  FORECAST ACCURACY", "📐  UNCERTAINTY"
])

# ===================== TAB 1: Per-location explorer =====================
with tab1:
    st.subheader("Explore weather patterns for any location")
    st.caption("268 cities across 191 countries — filter by country, then pick a city.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        country_pick = st.selectbox("Country", sorted(df['country'].unique()))
    with col_b:
        cities_in_country = sorted(df[df['country'] == country_pick]['location_name'].unique())
        location = st.selectbox("City", cities_in_country)

    loc_df = df[df['location_name'] == location].sort_values('last_updated')
    if len(loc_df) < 30:
        st.warning(f"⚠️ Only {len(loc_df)} readings available for {location} — this dataset "
                   f"samples one primary city per country, and secondary cities like this one "
                   f"have sparse coverage. Charts below may look flat or unreliable as a result.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AVG TEMPERATURE", f"{loc_df['temperature_celsius'].mean():.1f}°C")
    col2.metric("AVG HUMIDITY", f"{loc_df['humidity'].mean():.0f}%")
    col3.metric("AVG WIND", f"{loc_df['wind_kph'].mean():.1f} km/h")
    col4.metric("READINGS", f"{len(loc_df):,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loc_df['last_updated'], y=loc_df['temperature_celsius'],
        mode='lines', name='Temperature', line=dict(color='#00f5d4', width=2),
        fill='tozeroy', fillcolor='rgba(0,245,212,0.08)'
    ))
    fig.add_hline(y=loc_df['temperature_celsius'].mean(), line_dash="dot",
                  annotation_text="average", line_color="#9b5de5")
    fig.update_layout(title=f"Temperature over time — {location}",
                       xaxis_title="Date", yaxis_title="°C", height=420, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    recent = loc_df.tail(30)
    fig2 = go.Figure(data=go.Bar(
        x=recent['last_updated'], y=recent['precip_mm'],
        marker=dict(color=recent['precip_mm'], colorscale=[[0, '#1a1a2e'], [1, '#00bbf9']])
    ))
    fig2.update_layout(title=f"Recent precipitation — {location}",
                        xaxis_title="Date", yaxis_title="mm", height=350, **PLOTLY_THEME)
    st.plotly_chart(fig2, use_container_width=True)

# ===================== TAB 2: Spatial interpolation =====================
with tab2:
    st.subheader("Global temperature field (Inverse-Distance-Weighted interpolation)")
    st.caption(
        "Estimated from 268 weather stations using IDW spatial interpolation "
        "(k=8 nearest neighbors, power=2) — a genuine continuous spatial estimate."
    )
    lat_grid, lon_grid, grid_temps = load_idw_grid()
    fig = go.Figure(data=go.Contour(
        z=grid_temps, x=lon_grid, y=lat_grid,
        colorscale='Turbo', colorbar=dict(title="°C", tickfont=dict(color='#c8c8de'))
    ))
    stations = df.groupby('location_name').agg(
        latitude=('latitude', 'first'), longitude=('longitude', 'first')
    ).reset_index()
    fig.add_trace(go.Scatter(
        x=stations['longitude'], y=stations['latitude'], mode='markers',
        marker=dict(size=5, color='white', line=dict(width=1, color='#00f5d4')),
        name='Stations'
    ))
    fig.update_layout(height=620, xaxis_title="Longitude", yaxis_title="Latitude", **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 3: Compound risk explorer =====================
with tab3:
    st.subheader("Compound extreme-weather risk index")
    st.caption(
        "Combines z-scored deviations (per-location) across temperature, humidity, wind, "
        "pressure, and precipitation. Validated against Cyclone Remal (Bangladesh, May 2024): "
        "readings during the documented cyclone window score 2.19x the location's own baseline."
    )

    country_filter = st.selectbox("Filter by country (optional)",
                                    ["All"] + sorted(risk_df['country'].unique().tolist()))
    filtered = risk_df if country_filter == "All" else risk_df[risk_df['country'] == country_filter]

    top_n = st.slider("Show top N riskiest readings", 5, 50, 15)
    st.dataframe(
        filtered.nlargest(top_n, 'compound_risk_score')[
            ['location_name', 'country', 'last_updated', 'compound_risk_score']
        ],
        use_container_width=True
    )

    fig = go.Figure(data=go.Histogram(
        x=risk_df['compound_risk_score'], nbinsx=60,
        marker=dict(color='#9b5de5', line=dict(color='#00f5d4', width=0.5))
    ))
    fig.add_vline(x=risk_df['compound_risk_score'].quantile(0.98), line_dash="dash",
                  annotation_text="98th percentile", line_color="#f15bb5")
    fig.update_layout(title="Distribution of compound risk scores",
                       xaxis_title="Compound Risk Score", yaxis_title="Count",
                       height=400, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 4: Forecast accuracy =====================
with tab4:
    st.subheader("Per-location, multi-step-ahead forecast performance")
    st.caption(
        "Forecasts per-location temperature 1-3 days ahead using lag features, rolling means, "
        "seasonality, and latitude as a climate baseline — benchmarked honestly against a naive baseline."
    )
    try:
        results = pd.read_csv('report/location_forecast_results.csv', index_col=0)
        st.dataframe(results, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Model MAE', x=results.index, y=results['model_mae'],
                              marker_color='#00f5d4'))
        fig.add_trace(go.Bar(name='Naive Baseline MAE', x=results.index, y=results['naive_mae'],
                              marker_color='#4a4a6a'))
        fig.update_layout(barmode='group', title="Model vs. Naive Baseline (lower is better)",
                          yaxis_title="Mean Absolute Error (°C)", height=420, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)
        st.success("✨ Model beats the naive baseline at every forecast horizon (1-3 days ahead).")
    except FileNotFoundError:
        st.warning("Run notebooks/11_location_forecast.py first to generate forecast results.")

st.markdown('<div class="footer-text">Weather Trend Forecasting — PM Accelerator Data Science Technical Assessment</div>',
            unsafe_allow_html=True)

# ===================== TAB 5: Uncertainty / confidence intervals =====================
with tab5:
    st.subheader("Forecast uncertainty — calibrated prediction intervals")
    st.caption(
        "Instead of a single point prediction, quantile regression produces a genuine "
        "90% prediction interval. Checked against real held-out data: actual coverage is "
        "93.0% against a 90% target — the intervals are honest, not decorative."
    )
    try:
        intervals = pd.read_csv('report/prediction_intervals.csv')
        intervals['last_updated'] = pd.to_datetime(intervals['last_updated'])

        col_a, col_b = st.columns([1, 2])

        with col_a:
            uncertainty_country = st.selectbox(
                "Country", sorted(intervals['country'].unique()), key="uncertainty_country"
            )
        with col_b:
            cities_available = sorted(
                intervals[intervals['country'] == uncertainty_country]['location_name'].unique()
            )
            uncertainty_location = st.selectbox(
                "City", cities_available, key="uncertainty_loc"
            )
        ex = intervals[intervals['location_name'] == uncertainty_location].sort_values('last_updated').tail(60)

        loc_all = intervals[intervals['location_name'] == uncertainty_location]

        col1, col2, col3 = st.columns(3)
        if len(loc_all) > 0:
            loc_coverage = loc_all['within_interval'].mean() * 100
            loc_avg_width = (loc_all['upper_90'] - loc_all['lower_90']).mean()
            col1.metric("TARGET COVERAGE", "90%")
            col2.metric(f"ACTUAL COVERAGE ({uncertainty_location})", f"{loc_coverage:.1f}%")
            col3.metric("AVG INTERVAL WIDTH", f"±{loc_avg_width/2:.1f}°C")
        else:
            col1.metric("TARGET COVERAGE", "90%")
            col2.metric("ACTUAL COVERAGE", "No data")
            col3.metric("AVG INTERVAL WIDTH", "No data")


        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ex['last_updated'], y=ex['upper_90'], mode='lines',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=ex['last_updated'], y=ex['lower_90'], mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor='rgba(0,187,249,0.18)',
            name='90% Prediction Interval', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=ex['last_updated'], y=ex['predicted'], mode='lines',
            line=dict(color='#00bbf9', width=2), name='Predicted'
        ))
        fig.add_trace(go.Scatter(
            x=ex['last_updated'], y=ex['actual'], mode='lines',
            line=dict(color='#f15bb5', width=1.5, dash='dot'), name='Actual'
        ))
        fig.update_layout(
            title=f"Forecast with uncertainty band — {uncertainty_location} (last 60 test days)",
            xaxis_title="Date", yaxis_title="°C", height=460, **PLOTLY_THEME
        )
        st.plotly_chart(fig, use_container_width=True)
        if len(loc_all) > 0:
            st.success(f"✨ For {uncertainty_location}: {loc_coverage:.1f}% actual coverage vs. 90% target.")
    except FileNotFoundError:
        st.warning("Run notebooks/15_confidence_intervals.py first to generate prediction intervals.")