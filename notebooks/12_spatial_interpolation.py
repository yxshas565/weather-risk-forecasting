"""
Real spatial analysis: inverse-distance weighting (IDW) to estimate
temperature at unobserved coordinates from nearby weather stations,
instead of just groupby('country').mean() tables.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/cleaned_weather.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])

# Use each location's average temperature (across all its readings) as the
# "station value" for interpolation — a stable climate signal.
stations = (
    df.groupby('location_name')
    .agg(latitude=('latitude', 'first'),
         longitude=('longitude', 'first'),
         avg_temp=('temperature_celsius', 'mean'))
    .reset_index()
)
print(f"Stations used for interpolation: {len(stations)}")

def idw_interpolate(grid_lat, grid_lon, station_lats, station_lons, station_vals, power=2, k=8):
    """
    Inverse-distance weighting: estimates value at (grid_lat, grid_lon) using
    the k nearest stations, weighted by 1/distance^power. Standard spatial
    interpolation technique (simpler than kriging, doesn't require fitting a
    variogram, but captures the same core idea: nearby stations matter more).
    """
    # haversine-ish approx (good enough at this resolution): treat as
    # euclidean on lat/lon, adjust lon by cos(lat) to correct for convergence
    lat_rad = np.radians(grid_lat)
    dlat = station_lats - grid_lat
    dlon = (station_lons - grid_lon) * np.cos(lat_rad)
    dist = np.sqrt(dlat**2 + dlon**2)

    # avoid div-by-zero if grid point coincides with a station
    dist = np.where(dist == 0, 1e-6, dist)

    nearest_idx = np.argsort(dist)[:k]
    nearest_dist = dist[nearest_idx]
    nearest_val = station_vals[nearest_idx]

    weights = 1 / (nearest_dist ** power)
    return np.sum(weights * nearest_val) / np.sum(weights)

# Build a world grid and interpolate temperature at every grid point
lat_grid = np.arange(-60, 75, 2.5)
lon_grid = np.arange(-180, 180, 2.5)

station_lats = stations['latitude'].values
station_lons = stations['longitude'].values
station_vals = stations['avg_temp'].values

grid_temps = np.zeros((len(lat_grid), len(lon_grid)))
for i, lat in enumerate(lat_grid):
    for j, lon in enumerate(lon_grid):
        grid_temps[i, j] = idw_interpolate(lat, lon, station_lats, station_lons, station_vals)

print(f"Interpolated grid shape: {grid_temps.shape}")

# Visualize as a heatmap/contour
fig, ax = plt.subplots(figsize=(16, 8))
contour = ax.contourf(lon_grid, lat_grid, grid_temps, levels=20, cmap='RdYlBu_r')
ax.scatter(station_lons, station_lats, c='black', s=8, alpha=0.5, label='Weather stations')
plt.colorbar(contour, label='Interpolated Avg Temperature (°C)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Global Temperature Field — Inverse-Distance-Weighted Interpolation\n'
              f'({len(stations)} stations, k=8 nearest neighbors, power=2)')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('report/figures/spatial_idw_interpolation.png', dpi=150)
print("Saved report/figures/spatial_idw_interpolation.png")

# Save the grid itself for potential dashboard use
np.savez('data/idw_grid.npz', lat_grid=lat_grid, lon_grid=lon_grid, grid_temps=grid_temps)
print("Saved data/idw_grid.npz")