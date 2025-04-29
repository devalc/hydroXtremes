#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:56:49 2025

@author: cdeval
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os

# Create directory if it doesn't exist
output_dir = "./Results/summary_stats"
os.makedirs(output_dir, exist_ok=True)

drought_summary = pd.read_parquet("./Results/drought_intensity_magnitude_duration_summary.parquet")
flood_summary = pd.read_parquet("./Results/flood_intensity_magnitude_duration_summary.parquet")


#---------------------------------------------------------------------------------------------#
#Compute historical medians per station

# Filter historical floods
flood_hist = flood_summary[flood_summary['scenario'] == 'hist']

# Compute median values per station
flood_hist_medians = (
    flood_hist.groupby('station_name')
    .agg(
        duration_median_hist=('duration_weeks', 'median'),
        magnitude_median_hist=('magnitude', 'median'),
        intensity_median_hist=('intensity', 'median')
    )
    .reset_index()
)


# Save descriptive stats to a text file
with open('./historical_median_flood_metrics.txt', 'w') as f:
    f.write("--- Historical Median Flood Metrics Across Stations ---\n")

    metrics = {
        'duration_median_hist': 'Duration (weeks)',
        'magnitude_median_hist': 'Magnitude (cfs)',
        'intensity_median_hist': 'Intensity (cfs/week)'
    }

    for col, label in metrics.items():
        desc = flood_hist_medians[col].describe().round(2)
        f.write(f"\n{label}:\n")
        f.write(desc.to_string())
        f.write("\n")

#---------------------------------------------------------------------------------------------#


# Merge historical medians with all scenarios

# Merge with full flood dataset
flood_summary_norm = flood_summary.merge(flood_hist_medians, on='station_name', how='left')

#Normalize each metric relative to historical median

flood_summary_norm['duration_rel'] = (flood_summary_norm['duration_weeks'] / flood_summary_norm['duration_median_hist']-1)*100
flood_summary_norm['magnitude_rel'] = (flood_summary_norm['magnitude'] / flood_summary_norm['magnitude_median_hist']-1)*100
flood_summary_norm['intensity_rel'] = (flood_summary_norm['intensity'] / flood_summary_norm['intensity_median_hist']-1)*100


#---------------------------------------------------------------------------------------------#
# Compute absolute differences from historical median (per station)
#---------------------------------------------------------------------------------------------#
flood_summary_norm['duration_abs_delta'] = (
    flood_summary_norm['duration_weeks'] - flood_summary_norm['duration_median_hist']
)
flood_summary_norm['magnitude_abs_delta'] = (
    flood_summary_norm['magnitude'] - flood_summary_norm['magnitude_median_hist']
)
flood_summary_norm['intensity_abs_delta'] = (
    flood_summary_norm['intensity'] - flood_summary_norm['intensity_median_hist']
)

#---------------------------------------------------------------------------------------------#
# Combined summary: relative values with absolute delta in parentheses
#---------------------------------------------------------------------------------------------#
metrics = [
    ('duration_rel', 'duration_abs_delta'),
    ('magnitude_rel', 'magnitude_abs_delta'),
    ('intensity_rel', 'intensity_abs_delta')
]

# Define units for absolute differences
units = {
    'duration_rel': 'weeks',
    'magnitude_rel': 'cfs',
    'intensity_rel': 'cfs/week'
}

for rel_metric, abs_metric in metrics:
    unit = units[rel_metric]
    print(f"\n--- {rel_metric.upper()} ---")
    
    summary_rel = flood_summary_norm.groupby('scenario')[rel_metric].agg(
        median='median',
        # q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    
    summary_abs = flood_summary_norm.groupby('scenario')[abs_metric].agg(
        median='median',
        # q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    
    # Combine into single string column with (absolute) next to relative
    
    combined = (summary_rel.round(1).astype(str) + "% (" + 
    summary_abs.round(2).astype(str) + f" {unit})"
    )
    
    # Rename index columns
    combined.index.name = "Scenario"
    combined.columns = ['Median',  '75th Percentile']
    
    print(combined)

#---------------------------------------------------------------------------------------------#
# By scenario + time_period if more detailed breakdown is needed
#---------------------------------------------------------------------------------------------#

grouped = flood_summary_norm.groupby(['scenario', 'time_period'])

flood_metric_tables = {}

for rel_metric, abs_metric in metrics:
    unit = units[rel_metric]
    print(f"\n--- {rel_metric.upper()} BY SCENARIO & TIME PERIOD ---")
    
    rel_summary = grouped[rel_metric].quantile([0.5, 0.75]).unstack()
    abs_summary = grouped[abs_metric].quantile([0.5, 0.75]).unstack()
    
    rel_summary.columns = [f"{q}_rel" for q in ['q50', 'q75']]
    abs_summary.columns = [f"{q}_abs" for q in ['q50', 'q75']]
    
    merged = pd.concat([rel_summary, abs_summary], axis=1)
    
    # Combine relative and absolute into formatted string
    for q in ['q50', 'q75']:
        merged[q] = merged[f"{q}_rel"].round(0).astype(str) + "% (" + merged[f"{q}_abs"].round(3).astype(str) + f" {unit})"
    
    # Only keep formatted columns
    formatted = merged[['q50', 'q75']]
    formatted.columns = ['Median', '75th Percentile']
    
    formatted = formatted[
        ~formatted['Median'].str.contains('nan', case=False) &
        ~formatted['75th Percentile'].str.contains('nan', case=False)
    ]
    
    formatted = formatted.reset_index()
    
    metric_name = rel_metric.replace('_rel', '')
    
    flood_metric_tables[metric_name] = formatted
    
    print(formatted.reset_index())

#  Now save each table separately
for metric_name, table_df in flood_metric_tables.items():
    filename = os.path.join(output_dir, f"flood_summary_stats_{metric_name}.csv")
    table_df.to_csv(filename, index=False)
    print(f"Saved: {filename}")



#---------------------------------------------------------------------------------------------#
# seasonal analysis
#---------------------------------------------------------------------------------------------#

print("\n--- Historical Median Flood Metrics Across Stations by Season ---")

# Compute seasonal medians per station (if not already done)
flood_hist_seasonal_medians = (
    flood_summary[flood_summary['scenario'] == 'hist']
    .groupby(['station_name', 'season'])
    .agg(
        duration_median_hist=('duration_weeks', 'median'),
        magnitude_median_hist=('magnitude', 'median'),
        intensity_median_hist=('intensity', 'median')
    )
    .reset_index()
)

# Descriptive stats across stations, grouped by season
metrics = {
    'duration_median_hist': 'Duration (weeks)',
    'magnitude_median_hist': 'Magnitude (cfs)',
    'intensity_median_hist': 'Intensity (cfs/week)'
}

for col, label in metrics.items():
    print(f"\n--- {label} ---")
    desc_by_season = (
        flood_hist_seasonal_medians
        .groupby('season')[col]
        .describe()
        .round(2)
    )
    print(desc_by_season)

# Merge seasonal historical medians with the full dataset
flood_summary_norm_seasonal = flood_summary.merge(
    flood_hist_seasonal_medians,
    on=['station_name', 'season'],
    how='left'
)

# Normalize each metric relative to seasonal historical median
flood_summary_norm_seasonal['duration_rel'] = (flood_summary_norm_seasonal['duration_weeks'] / flood_summary_norm_seasonal['duration_median_hist'] - 1)*100
flood_summary_norm_seasonal['magnitude_rel'] = (flood_summary_norm_seasonal['magnitude'] / flood_summary_norm_seasonal['magnitude_median_hist']- 1)*100
flood_summary_norm_seasonal['intensity_rel'] = (flood_summary_norm_seasonal['intensity'] / flood_summary_norm_seasonal['intensity_median_hist']- 1)*100

# Compute absolute deltas
flood_summary_norm_seasonal['duration_abs_delta'] = (
    flood_summary_norm_seasonal['duration_weeks'] - flood_summary_norm_seasonal['duration_median_hist']
)
flood_summary_norm_seasonal['magnitude_abs_delta'] = (
    flood_summary_norm_seasonal['magnitude'] - flood_summary_norm_seasonal['magnitude_median_hist']
)
flood_summary_norm_seasonal['intensity_abs_delta'] = (
    flood_summary_norm_seasonal['intensity'] - flood_summary_norm_seasonal['intensity_median_hist']
)

#---------------------------------------------------------------------------------------------#
# Updated Seasonal Analysis with Relative + Absolute Combined
metric_units = {
    'duration_rel': 'weeks',
    'magnitude_rel': 'cfs',
    'intensity_rel': 'cfs/week'
}

metrics = [
    ('duration_rel', 'duration_abs_delta'),
    ('magnitude_rel', 'magnitude_abs_delta'),
    ('intensity_rel', 'intensity_abs_delta')
]

grouped_seasonal = flood_summary_norm_seasonal.groupby(['scenario', 'time_period', 'season'])


# Dictionary to collect seasonal tables
seasonal_metric_tables = {}

for rel_metric, abs_metric in metrics:
    print(f"\n--- {rel_metric.upper()} BY SCENARIO, TIME PERIOD & SEASON ---")
    
    unit = metric_units[rel_metric]  # Pick correct unit

    # Compute relative and absolute quartiles
    rel_summary = grouped_seasonal[rel_metric].quantile([0.5, 0.75]).unstack()
    abs_summary = grouped_seasonal[abs_metric].quantile([0.5, 0.75]).unstack()

    # Rename columns
    rel_summary.columns = [f"{q}_rel" for q in ['q50', 'q75']]
    abs_summary.columns = [f"{q}_abs" for q in ['q50', 'q75']]

    # Merge summaries
    merged = pd.concat([rel_summary, abs_summary], axis=1)

    # Format combined relative and absolute
    for q in ['q50', 'q75']:
        merged[q] = (merged[f"{q}_rel"]).round(0).astype(str) + "% (" + merged[f"{q}_abs"].round(4).astype(str) + f" {unit})"

    # Keep only formatted columns
    formatted = merged[['q50', 'q75']].reset_index()
    formatted.columns = ['scenario', 'time_period', 'season', 'Median', '75th Percentile']

    # Drop rows with 'nan'
    formatted = formatted[
        ~formatted['Median'].str.contains('nan', case=False) &
        ~formatted['75th Percentile'].str.contains('nan', case=False)
    ]

    # Save table in dictionary
    metric_name = rel_metric.replace('_rel', '')  # 'duration', 'magnitude', etc.
    seasonal_metric_tables[metric_name] = formatted

    print(formatted)

# Now save each table to CSV
for metric_name, table_df in seasonal_metric_tables.items():
    filename = os.path.join(output_dir, f"flood_seasonal_summary_stats_{metric_name}.csv")
    table_df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

#---------------------------------------------------------------------------------------------#
#DROUGHTS
#---------------------------------------------------------------------------------------------#
#Compute historical medians per station

# Filter historical droughts
drought_hist = drought_summary[drought_summary['scenario'] == 'hist']

# Compute median values per station
drought_hist_medians = (
    drought_hist.groupby('station_name')
    .agg(
        duration_median_hist=('duration_weeks', 'median'),
        magnitude_median_hist=('magnitude', 'median'),
        intensity_median_hist=('intensity', 'median')
    )
    .reset_index()
)



# Descriptive stats for historical median values (across stations)
print("\n--- Historical Median Drought Metrics Across Stations ---")

metrics = {
    'duration_median_hist': 'Duration (weeks)',
    'magnitude_median_hist': 'Magnitude (cfs)',
    'intensity_median_hist': 'Intensity (cfs/week)'
}

for col, label in metrics.items():
    desc = drought_hist_medians[col].describe().round(2)
    print(f"\n{label}:")
    print(desc)


#---------------------------------------------------------------------------------------------#

# Merge historical medians with all scenarios

# Merge with full drought dataset
drought_summary_norm = drought_summary.merge(drought_hist_medians, on='station_name', how='left')

#Normalize each metric relative to historical median

drought_summary_norm['duration_rel'] = (drought_summary_norm['duration_weeks'] / drought_summary_norm['duration_median_hist'] - 1) * 100
drought_summary_norm['magnitude_rel'] = (drought_summary_norm['magnitude'] / drought_summary_norm['magnitude_median_hist'] - 1) * 100
drought_summary_norm['intensity_rel'] = (drought_summary_norm['intensity'] / drought_summary_norm['intensity_median_hist'] - 1) * 100


#---------------------------------------------------------------------------------------------#
# Compute absolute differences from historical median (per station)
#---------------------------------------------------------------------------------------------#
drought_summary_norm['duration_abs_delta'] = (
    drought_summary_norm['duration_weeks'] - drought_summary_norm['duration_median_hist']
)
drought_summary_norm['magnitude_abs_delta'] = (
    drought_summary_norm['magnitude'] - drought_summary_norm['magnitude_median_hist']
)
drought_summary_norm['intensity_abs_delta'] = (
    drought_summary_norm['intensity'] - drought_summary_norm['intensity_median_hist']
)


#---------------------------------------------------------------------------------------------#
# Combined summary: relative values with absolute delta in parentheses
#---------------------------------------------------------------------------------------------#
metrics = [
    ('duration_rel', 'duration_abs_delta'),
    ('magnitude_rel', 'magnitude_abs_delta'),
    ('intensity_rel', 'intensity_abs_delta')
]

# Define units for absolute differences
units = {
    'duration_rel': 'weeks',
    'magnitude_rel': 'cfs',
    'intensity_rel': 'cfs/week'
}

for rel_metric, abs_metric in metrics:
    unit = units[rel_metric]
    print(f"\n--- {rel_metric.upper()} ---")
    
    d_summary_rel = drought_summary_norm.groupby('scenario')[rel_metric].agg(
        median='median',
        q75=lambda x: x.quantile(0.75)
    )
    
    d_summary_abs = drought_summary_norm.groupby('scenario')[abs_metric].agg(
        median='median',
        q75=lambda x: x.quantile(0.75)
    )
    
    # Combine into single string column with (absolute) next to relative
    d_combined = (
        d_summary_rel.round(1).astype(str) + "% (" +
        d_summary_abs.round(2).astype(str) + f" {unit})"
    )
    
    # Rename index columns
    d_combined.index.name = "Scenario"
    d_combined.columns = ['Median',  '75th Percentile']
    
    print(d_combined)

#---------------------------------------------------------------------------------------------#
# By scenario + time_period if more detailed breakdown is needed
#---------------------------------------------------------------------------------------------#

grouped = drought_summary_norm.groupby(['scenario', 'time_period'])

metric_tables = {}

for rel_metric, abs_metric in metrics:
    unit = units[rel_metric]
    print(f"\n--- {rel_metric.upper()} BY SCENARIO & TIME PERIOD ---")
    
    d_rel_summary = grouped[rel_metric].quantile([0.5, 0.75]).unstack()
    d_abs_summary = grouped[abs_metric].quantile([0.5, 0.75]).unstack()
    
    d_rel_summary.columns = [f"{q}_rel" for q in ['q50', 'q75']]
    d_abs_summary.columns = [f"{q}_abs" for q in ['q50', 'q75']]
    
    d_merged = pd.concat([d_rel_summary, d_abs_summary], axis=1)
    
    # Combine relative and absolute into formatted string
    for q in ['q50', 'q75']:
        d_merged[q] = d_merged[f"{q}_rel"].round(0).astype(str) + " % (" + d_merged[f"{q}_abs"].round(3).astype(str) + f" {unit})"
    
    # Only keep formatted columns
    d_formatted = d_merged[['q50', 'q75']]
    d_formatted.columns = ['Median', '75th Percentile']
    
    d_formatted = d_formatted[
        ~d_formatted['Median'].str.contains('nan', case=False) &
        ~d_formatted['75th Percentile'].str.contains('nan', case=False)
    ]
    
    d_formatted = d_formatted.reset_index()
    
    metric_name = rel_metric.replace('_rel', '')
    
    metric_tables[metric_name] = d_formatted
    
    print(d_formatted.reset_index())

#  Now save each table separately
for metric_name, table_df in metric_tables.items():
    filename = os.path.join(output_dir, f"drought_summary_stats_{metric_name}.csv")
    table_df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

#---------------------------------------------------------------------------------------------#
# seasonal drought analysis
#---------------------------------------------------------------------------------------------#


print("\n--- Historical Median Drought Metrics Across Stations by Season ---")

# Compute seasonal medians per station (if not already done)
drought_hist_seasonal_medians = (
    drought_summary[drought_summary['scenario'] == 'hist']
    .groupby(['station_name', 'season'])
    .agg(
        duration_median_hist=('duration_weeks', 'median'),
        magnitude_median_hist=('magnitude', 'median'),
        intensity_median_hist=('intensity', 'median')
    )
    .reset_index()
)

# Descriptive stats across stations, grouped by season
metrics = {
    'duration_median_hist': 'Duration (weeks)',
    'magnitude_median_hist': 'Magnitude (cfs)',
    'intensity_median_hist': 'Intensity (cfs/week)'
}

for col, label in metrics.items():
    print(f"\n--- {label} ---")
    d_desc_by_season = (
        drought_hist_seasonal_medians
        .groupby('season')[col]
        .describe()
        .round(2)
    )
    print(d_desc_by_season)

# Merge seasonal historical medians with the full dataset
drought_summary_norm_seasonal = drought_summary.merge(
    drought_hist_seasonal_medians,
    on=['station_name', 'season'],
    how='left'
)

# Normalize each metric relative to seasonal historical median
drought_summary_norm_seasonal['duration_rel'] = (drought_summary_norm_seasonal['duration_weeks'] / drought_summary_norm_seasonal['duration_median_hist'] -1)*100
drought_summary_norm_seasonal['magnitude_rel'] = (drought_summary_norm_seasonal['magnitude'] / drought_summary_norm_seasonal['magnitude_median_hist']-1)*100
drought_summary_norm_seasonal['intensity_rel'] = (drought_summary_norm_seasonal['intensity'] / drought_summary_norm_seasonal['intensity_median_hist']-1)*100

# Compute absolute deltas
drought_summary_norm_seasonal['duration_abs_delta'] = (
    drought_summary_norm_seasonal['duration_weeks'] - drought_summary_norm_seasonal['duration_median_hist']
)
drought_summary_norm_seasonal['magnitude_abs_delta'] = (
    drought_summary_norm_seasonal['magnitude'] - drought_summary_norm_seasonal['magnitude_median_hist']
)
drought_summary_norm_seasonal['intensity_abs_delta'] = (
    drought_summary_norm_seasonal['intensity'] - drought_summary_norm_seasonal['intensity_median_hist']
)

#---------------------------------------------------------------------------------------------#

# Seasonal Analysis — Formatted Output

# Define units for the metrics
metric_units = {
    'duration_rel': 'weeks',
    'magnitude_rel': 'cfs',
    'intensity_rel': 'cfs/week'
}

metrics = [
    ('duration_rel', 'duration_abs_delta'),
    ('magnitude_rel', 'magnitude_abs_delta'),
    ('intensity_rel', 'intensity_abs_delta')
]

grouped_seasonal = drought_summary_norm_seasonal.groupby(['scenario', 'time_period', 'season'])

# Dictionary to collect seasonal tables
seasonal_metric_tables = {}

for rel_metric, abs_metric in metrics:
    print(f"\n--- {rel_metric.upper()} BY SCENARIO, TIME PERIOD & SEASON ---")
    
    unit = metric_units[rel_metric]  # Pick correct unit

    # Compute relative and absolute quartiles
    d_rel_summary = grouped_seasonal[rel_metric].quantile([0.5, 0.75]).unstack()
    d_abs_summary = grouped_seasonal[abs_metric].quantile([0.5, 0.75]).unstack()

    # Rename columns
    d_rel_summary.columns = [f"{q}_rel" for q in ['q50', 'q75']]
    d_abs_summary.columns = [f"{q}_abs" for q in ['q50', 'q75']]

    # Merge summaries
    d_merged = pd.concat([d_rel_summary, d_abs_summary], axis=1)

    # Format combined relative and absolute
    for q in ['q50', 'q75']:
        d_merged[q] = (d_merged[f"{q}_rel"]).round(0).astype(str) + "% (" + d_merged[f"{q}_abs"].round(4).astype(str) + f" {unit})"

    # Keep only formatted columns
    d_formatted = d_merged[['q50', 'q75']].reset_index()
    d_formatted.columns = ['scenario', 'time_period', 'season', 'Median', '75th Percentile']

    # Drop rows with 'nan'
    d_formatted = d_formatted[
        ~d_formatted['Median'].str.contains('nan', case=False) &
        ~d_formatted['75th Percentile'].str.contains('nan', case=False)
    ]

    # Save table in dictionary
    metric_name = rel_metric.replace('_rel', '')  # 'duration', 'magnitude', etc.
    seasonal_metric_tables[metric_name] = d_formatted

    print(d_formatted)

# Now save each table to CSV
for metric_name, table_df in seasonal_metric_tables.items():
    filename = os.path.join(output_dir, f"drought_seasonal_summary_stats_{metric_name}.csv")
    table_df.to_csv(filename, index=False)
    print(f"Saved: {filename}")
