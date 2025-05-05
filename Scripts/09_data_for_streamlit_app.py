#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:34:52 2025

@author: cdeval
"""

import pandas as pd
import os

# Output directory for Streamlit app
outdir =  './app/data/'
os.makedirs(outdir, exist_ok=True)

# Load data
drought_summary = pd.read_parquet("./Results/drought_intensity_magnitude_duration_summary.parquet")
flood_summary = pd.read_parquet("./Results/flood_intensity_magnitude_duration_summary.parquet")

# Define columns
group_cols = ['station_name', 'scenario', 'time_period', 'season']
median_cols = ['duration_weeks', 'magnitude', 'intensity']
meta_cols = [
    'wmo_reg', 'sub_reg', 'river', 'station', 'country', 'lat', 'long',
    'area', 'Name', 'Continent', 'Type', 'Acronym'
]

# Group and compute median with warning-suppressing option
grouped_drought = drought_summary.groupby(group_cols, observed=False)[median_cols].median().reset_index()
grouped_flood = flood_summary.groupby(group_cols, observed=False)[median_cols].median().reset_index()

# Filter invalid scenario/time_period combinations
filtered_drought = grouped_drought[
    ~(
        ((grouped_drought['scenario'] == 'hist') & (grouped_drought['time_period'] != 'historical')) |
        ((grouped_drought['scenario'] != 'hist') & (grouped_drought['time_period'] == 'historical'))
    )
]


# Define replacements flags for no event
fill = {
    'intensity': -9999.0,
    'magnitude': -9999.0,
    'duration_weeks': -9999.0
}

# Apply fill to  DataFrame
filtered_drought = filtered_drought.fillna(value={**fill})

filtered_flood = grouped_flood[
    ~(
        ((grouped_flood['scenario'] == 'hist') & (grouped_flood['time_period'] != 'historical')) |
        ((grouped_flood['scenario'] != 'hist') & (grouped_flood['time_period'] == 'historical'))
    )
]

# Apply fill to  DataFrame
filtered_flood = filtered_flood.fillna(value={**fill})


# Get metadata for each unique station from both drought and flood datasets
station_meta = pd.concat([
    drought_summary[['station_name'] + meta_cols],
    flood_summary[['station_name'] + meta_cols]
]).drop_duplicates(subset='station_name')



# Rename metric columns for clarity
result_drought = filtered_drought.rename(columns={
    'duration_weeks': 'drought_duration_weeks',
    'magnitude': 'drought_magnitude',
    'intensity': 'drought_intensity'
})

result_flood = filtered_flood.rename(columns={
    'duration_weeks': 'flood_duration_weeks',
    'magnitude': 'flood_magnitude',
    'intensity': 'flood_intensity'
})

# Merge drought and flood summaries on common group columns
final_df = pd.merge(result_drought, result_flood, on=group_cols, how='outer')

# Attach metadata based on station_name only (metadata is unique per station)
final_df = pd.merge(final_df, station_meta, on='station_name', how='left')

# Save final result to Parquet or CSV
final_df.to_parquet(os.path.join(outdir, "drought_flood_IDM_combined_with_metadata.parquet"), index=False)
