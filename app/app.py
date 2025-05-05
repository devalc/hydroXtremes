#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:30:00 2025
@author: cdeval
"""

import pandas as pd
import plotly.express as px
import streamlit as st
import geopandas as gpd

# App page configuration
st.set_page_config(
    page_title="FlowView: Visualizing Extreme Hydrological Events",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the combined drought/flood data with metadata
file_path = 'app/data/drought_flood_IDM_combined_with_metadata.parquet'
data = pd.read_parquet(file_path)

# Load IPCC region boundaries
parquet_file_path = 'app/data/IPCC-WGI-reference-regions-v4.parquet'
parquet_data = gpd.read_parquet(parquet_file_path)
parquet_data = parquet_data.set_crs('EPSG:4326', allow_override=True)

# App title
st.title("FlowView: Visualizing Extreme Hydrological Events")

# Sidebar controls
st.sidebar.header("Filter Options")

event_type = st.sidebar.selectbox("Select Event Type:", ['Drought', 'Flood'])
metric = st.sidebar.selectbox("Select Variable to Display:", ['Duration', 'Magnitude', 'Intensity'])
scenario_options = data['scenario'].unique()
selected_scenario = st.sidebar.selectbox("Select Scenario:", scenario_options)

# Time period options
if selected_scenario == 'hist':
    time_period_options = ['historical']
else:
    time_period_options = [period for period in data['time_period'].unique() if period != 'historical']
selected_time_period = st.sidebar.selectbox("Select Time Period:", time_period_options)

# Season options
season_options = data['season'].unique()
selected_season = st.sidebar.selectbox("Select Season:", season_options)

# Mapping for metric columns
metric_map = {
    'Drought': {
        'Duration': 'drought_duration_weeks',
        'Magnitude': 'drought_magnitude',
        'Intensity': 'drought_intensity'
    },
    'Flood': {
        'Duration': 'flood_duration_weeks',
        'Magnitude': 'flood_magnitude',
        'Intensity': 'flood_intensity'
    }
}
selected_column = metric_map[event_type][metric]

# Round values in main dataset for consistency
data[selected_column] = data[selected_column].round(2)

# Global binning across all valid data
flag_value = -9999.0
global_valid_data = data[data[selected_column] != flag_value].copy()

try:
    global_bins = pd.qcut(global_valid_data[selected_column], q=4, duplicates='drop')
    bin_intervals = global_bins.cat.categories
    bin_edges = [interval.left for interval in bin_intervals] + [bin_intervals[-1].right]
    bin_labels = [f"{round(interval.left, 2)}–{round(interval.right, 2)}" for interval in bin_intervals]
except ValueError:
    bin_edges = None
    bin_labels = None

# Filter data based on sidebar selections
filtered_data = data[
    (data['scenario'] == selected_scenario) &
    (data['time_period'] == selected_time_period) &
    (data['season'] == selected_season)
].copy()

# Separate data with and without events
no_event_data = filtered_data[filtered_data[selected_column] == flag_value].copy()
event_data = filtered_data[filtered_data[selected_column] != flag_value].copy()

# Apply consistent binning to filtered data
if bin_edges and bin_labels:
    event_data['category'] = pd.cut(
        event_data[selected_column],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True
    )
else:
    event_data['category'] = 'Single Bin'

# Label no-event data
no_event_data['category'] = 'No Event'

# Combine datasets
final_data = pd.concat([event_data, no_event_data], ignore_index=True)

# Set category order
if isinstance(event_data['category'].dtype, pd.CategoricalDtype):
    category_order = list(event_data['category'].cat.categories) + ['No Event']
else:
    category_order = ['Single Bin', 'No Event']

# Title
st.subheader(f"{event_type} {metric} Map – {selected_scenario.upper()}, {selected_time_period}, {selected_season}")

# Color Scheme: Diverging color palette for event categories, gray for 'No Event'
colors = px.colors.diverging.Portland[:len(category_order)-1] + ['gray']

# Plotting
fig = px.scatter_mapbox(
    final_data,
    lat='lat',
    lon='long',
    color='category',
    hover_data={
        selected_column: ':.2f',  # Rounded hover data
        'station': True,
        'river': True,
        'country': True,
        'station_name': True
    },
    zoom=3,
    center={"lat": 39.8283, "lon": -98.5795},  # Center on CONUS
    height=800,
    category_orders={'category': category_order},
    color_discrete_sequence=colors,  # Apply the custom color scheme,

)

# Add IPCC polygons
for _, row in parquet_data.iterrows():
    geometry = row['geometry']
    if geometry.geom_type == 'Polygon':
        coords = geometry.exterior.coords
        lon, lat = zip(*coords)
        fig.add_scattermapbox(
            mode = 'lines',
            fill="none",
            line=dict(color="black", width=1),
            lon=lon,
            lat=lat,
            hovertext=row['Name'],
            showlegend=False
        )
    elif geometry.geom_type == 'MultiPolygon':
        for poly in geometry.geoms:
            coords = poly.exterior.coords
            lon, lat = zip(*coords)
            fig.add_scattermapbox(
                mode = 'lines',
                fill="none",
                line=dict(color="black", width=1),
                lon=lon,
                lat=lat,
                hovertext=row['Name'],
                showlegend=False
            )

# Layout updates
fig.update_traces(marker=dict(size=10))
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(legend_title_text=f"{event_type} {metric} Range")

# Display plot
st.plotly_chart(fig, use_container_width=True)