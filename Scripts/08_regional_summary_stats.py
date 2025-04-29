#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:40:34 2025

regional summaries

@author: cdeval
"""

#---------------------------------------------------------------------------------------------#
import pandas as pd
import geopandas as gpd
import os

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

# Merge historical medians with all scenarios

# Merge with full flood dataset
flood_summary_norm = flood_summary.merge(flood_hist_medians, on='station_name', how='left')

#Normalize each metric relative to historical median

flood_summary_norm['duration_rel'] = (flood_summary_norm['duration_weeks'] / flood_summary_norm['duration_median_hist']-1)*100
flood_summary_norm['magnitude_rel'] = (flood_summary_norm['magnitude'] / flood_summary_norm['magnitude_median_hist']-1)*100
flood_summary_norm['intensity_rel'] = (flood_summary_norm['intensity'] / flood_summary_norm['intensity_median_hist']-1)*100


#---------------------------------------------------------------------------------------------#

srex_gdf = gpd.read_file("./data/IPCC-WGI-reference-regions-v4_shapefile/IPCC-WGI-reference-regions-v4.shp")
srex_gdf = srex_gdf.to_crs('EPSG:4326')

# Aggregate median relative flood metrics by SREX region and scenario
flood_region_summary = (
    flood_summary_norm
    .groupby(['Name', 'scenario'])
    .agg(
        rel_duration_median=('duration_rel', 'median'),
        rel_magnitude_median=('magnitude_rel', 'median'),
        rel_intensity_median=('intensity_rel', 'median')
    )
    .reset_index()
)

flood_region_summary = flood_region_summary[flood_region_summary['scenario'] != 'hist']

#---------------------------------------------------------------------------------------------#
# Regional summaries      

def summarize_top_bottom(df, metric, output_dir='Results/summary_stats/region_summaries'):
    os.makedirs(output_dir, exist_ok=True)

    for scenario in ['hist', 'rcp4p5', 'rcp8p5']:
        scenario_label = scenario.upper()

        # Top 5
        top5 = (
            df[df['scenario'] == scenario]
            .sort_values(f'rel_{metric}_median', ascending=False)
            [['Name', f'rel_{metric}_median']]
            .head(5)
            .round(1)
            .rename(columns={f'rel_{metric}_median': f'{metric}_percent_change_from_hist'})
        )
        print(f"\nTop 5 regions by flood {metric} - {scenario_label} (percent change from historical)")
        print(top5.to_string(index=False))
        top5.to_csv(f"{output_dir}/top5_{metric}_{scenario}.csv", index=False)

        # Bottom 5
        bottom5 = (
            df[df['scenario'] == scenario]
            .sort_values(f'rel_{metric}_median', ascending=True)
            [['Name', f'rel_{metric}_median']]
            .head(5)
            .round(1)
            .rename(columns={f'rel_{metric}_median': f'{metric}_percent_change_from_hist'})
        )
        print(f"\nBottom 5 regions by flood {metric} - {scenario_label} (percent change from historical)")
        print(bottom5.to_string(index=False))
        bottom5.to_csv(f"{output_dir}/bottom5_{metric}_{scenario}.csv", index=False)


# Run for magnitude and duration

summarize_top_bottom(flood_region_summary, 'duration')
summarize_top_bottom(flood_region_summary, 'magnitude')
summarize_top_bottom(flood_region_summary, 'intensity')


#---------------------------------------------------------------------------------------------#
#Drought  Regional summaries
#---------------------------------------------------------------------------------------------#


# Filter historical droughts
drought_hist = drought_summary[drought_summary['scenario'] == 'hist']

# Compute median values per station for droughts
drought_hist_medians = (
    drought_hist.groupby('station_name')
    .agg(
        duration_median_hist=('duration_weeks', 'median'),
        magnitude_median_hist=('magnitude', 'median'),
        intensity_median_hist=('intensity', 'median')
    )
    .reset_index()
)

# Merge with full drought dataset
drought_summary_norm = drought_summary.merge(drought_hist_medians, on='station_name', how='left')

# Normalize each metric relative to historical median
drought_summary_norm['duration_rel'] = (drought_summary_norm['duration_weeks'] / drought_summary_norm['duration_median_hist']-1)*100
drought_summary_norm['magnitude_rel'] = (drought_summary_norm['magnitude'] / drought_summary_norm['magnitude_median_hist']-1)*100
drought_summary_norm['intensity_rel'] = (drought_summary_norm['intensity'] / drought_summary_norm['intensity_median_hist']-1)*100


# Aggregate median relative drought metrics by SREX region and scenario
drought_region_summary = (
    drought_summary_norm
    .groupby(['Name', 'scenario'])
    .agg(
        rel_duration_median=('duration_rel', 'median'),
        rel_magnitude_median=('magnitude_rel', 'median'),
        rel_intensity_median=('intensity_rel', 'median')
    )
    .reset_index()
)

# Drop hist for future projections only
drought_region_summary = drought_region_summary[drought_region_summary['scenario'] != 'hist']



def summarize_top_bottom(df, metric, output_dir='Results/summary_stats/region_summaries_drought'):
    os.makedirs(output_dir, exist_ok=True)

    for scenario in ['hist', 'rcp4p5', 'rcp8p5']:
        scenario_label = scenario.upper()

        # Top 5
        top5 = (
            df[df['scenario'] == scenario]
            .sort_values(f'rel_{metric}_median', ascending=False)
            [['Name', f'rel_{metric}_median']]
            .head(5)
            .round(1)
            .rename(columns={f'rel_{metric}_median': f'{metric}_percent_change_from_hist'})
        )
        print(f"\nTop 5 regions by drought {metric} - {scenario_label} (percent change from historical)")
        print(top5.to_string(index=False))
        top5.to_csv(f"{output_dir}/top5_{metric}_{scenario}.csv", index=False)

        # Bottom 5
        bottom5 = (
            df[df['scenario'] == scenario]
            .sort_values(f'rel_{metric}_median', ascending=True)
            [['Name', f'rel_{metric}_median']]
            .head(5)
            .round(1)
            .rename(columns={f'rel_{metric}_median': f'{metric}_percent_change_from_hist'})
        )
        print(f"\nBottom 5 regions by drought {metric} - {scenario_label} (percent change from historical)")
        print(bottom5.to_string(index=False))
        bottom5.to_csv(f"{output_dir}/bottom5_{metric}_{scenario}.csv", index=False)


# Run for magnitude and duration

summarize_top_bottom(drought_region_summary, 'duration')
summarize_top_bottom(drought_region_summary, 'magnitude')
summarize_top_bottom(drought_region_summary, 'intensity')
