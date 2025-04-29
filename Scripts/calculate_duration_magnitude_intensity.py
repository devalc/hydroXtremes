#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:31:15 2025

Calculate_magnitude, intensity and duration of floods and droughts

@author: cdeval
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import geopandas as gpd
from shapely.geometry import Point

#---------------------------------------------------------------------------------------------#
# join station metadata
station_info = pd.read_csv("./data/staion_id.csv")
station_info.head()

# Create GeoDataFrame
station_gdf = gpd.GeoDataFrame(
    station_info,
    geometry=gpd.points_from_xy(station_info['long'], station_info['lat']),
    crs='EPSG:4326'  # WGS84
)
station_gdf.head()

#SREX regions
srex_gdf = gpd.read_file("./data/IPCC-WGI-reference-regions-v4_shapefile/IPCC-WGI-reference-regions-v4.shp")
srex_gdf = srex_gdf.to_crs('EPSG:4326')
srex_gdf.head()

srex_gdf['geometry'] = srex_gdf['geometry'].buffer(0.001)

#spatial join, assign srex regions
station_gdf_with_srex = gpd.sjoin(station_gdf, srex_gdf[['Name', 'Continent','Type','Acronym', 'geometry']], how='left', predicate='within').drop(columns='index_right')

station_gdf_with_srex.head()

#---------------------------------------------------------------------------------------------#

# Specify the path to the extracted Parquet file
parquet_file = "./Results/grouped_median_discharge.parquet"

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file)


df.head()
#df.shape


df['scenario'].unique()

# First, find stations that have any NaNs
stations_with_nan = df[df['discharge'].isna()]['station_name'].unique()
len(stations_with_nan)


# Then, remove all rows for those stations
df_clean = df[~df['station_name'].isin(stations_with_nan)].copy()

print("Stations before:", df['station_name'].nunique())
print("Stations after:", df_clean['station_name'].nunique())


# Ensure time is in datetime format
df_clean['time'] = pd.to_datetime(df_clean['time'])

# Extract year from time column
df_clean['year'] = df_clean['time'].dt.year

# Group by scenario and get min and max year
year_range = df_clean.groupby('scenario')['year'].agg(['min', 'max'])

print(year_range)


# Add time period labels with the new desired periods
df_clean['time_period'] = pd.cut(
    df_clean['year'],
    bins=[1975, 2005, 2039, 2069, 2099],  # Adjusting the bin ranges
    labels=['historical', 'early', 'mid', 'end'],
    right=True  # Include the upper bound, so 2005 goes to 'historical' and 2039 to 'early'
)


# Extract month for seasonal grouping
df_clean['month'] = df_clean['time'].dt.month

# Add seasonal labels
season_conditions = [
    df_clean['month'].isin([12, 1, 2]),
    df_clean['month'].isin([3, 4, 5]),
    df_clean['month'].isin([6, 7, 8]),
    df_clean['month'].isin([9, 10, 11])
]
season_choices = ['DJF', 'MAM', 'JJA', 'SON']
df_clean['season'] = np.select(season_conditions, season_choices)

#---------------------------------------------------------------------------------------------#
# Filter for historical scenario
hist_df = df_clean[df_clean['scenario'] == 'hist']

# Group by station and calculate percentiles
percentiles = hist_df.groupby('station_name')['discharge'].quantile([0.01, 0.99]).unstack()
percentiles.columns = ['p01', 'p99']


# Merge thresholds into the full DataFrame
df_clean = df_clean.merge(percentiles, on='station_name', how='left')

# Flag drought and flood weeks
df_clean['is_drought'] = df_clean['discharge'] < df_clean['p01']
df_clean['is_flood'] = df_clean['discharge'] > df_clean['p99']


# Look at a sample
print(df_clean[['station_name', 'scenario', 'time', 'discharge', 'p01', 'p99', 'is_drought', 'is_flood']].head())

#---------------------------------------------------------------------------------------------#


group_cols = ['station_name', 'scenario']

df_clean = df_clean.sort_values(by=['station_name', 'scenario', 'time'])

# Identify where a new drought starts
df_clean['new_drought_event'] = (
    (df_clean['is_drought'] != df_clean.groupby(group_cols)['is_drought'].shift())
    & df_clean['is_drought']
)

# Cumulative sum of events per group
df_clean['drought_event_id'] = df_clean.groupby(group_cols)['new_drought_event'].cumsum()

# Mask non-drought weeks
df_clean.loc[~df_clean['is_drought'], 'drought_event_id'] = np.nan

# Repeat for flood
df_clean['new_flood_event'] = (
    (df_clean['is_flood'] != df_clean.groupby(group_cols)['is_flood'].shift())
    & df_clean['is_flood']
)

df_clean['flood_event_id'] = df_clean.groupby(group_cols)['new_flood_event'].cumsum()
df_clean.loc[~df_clean['is_flood'], 'flood_event_id'] = np.nan

#---------------------------------------------------------------------------------------------#

# Filter drought rows and calculate deficit
df_drought = df_clean[df_clean['is_drought']].copy()
df_drought['deficit'] = df_drought['p01'] - df_drought['discharge']
# df_drought = df_drought[0:10000]

# Group and summarize
drought_summary = (
    df_drought.groupby(['station_name', 'scenario', 'time_period', 'season', 'drought_event_id'], observed = True)
    .agg(
        start_time=('time', 'min'),
        end_time=('time', 'max'),
        duration_weeks=('time', 'count'),
        magnitude=('deficit', 'mean'),
    )
    .reset_index()
)

drought_summary['intensity'] = drought_summary['magnitude'] / drought_summary['duration_weeks']

# Ensure station_name and event_ids are integer
drought_summary['station_name'] = drought_summary['station_name'].astype(int)
drought_summary['drought_event_id'] = drought_summary['drought_event_id'].astype(int)


# Round magnitude and intensity
drought_summary['magnitude'] = drought_summary['magnitude'].round(2)
drought_summary['intensity'] = drought_summary['intensity'].round(2)


# Merge station metadata into drought summaries
drought_summary = drought_summary.merge(station_gdf_with_srex, on='station_name', how='left')
drought_summary = drought_summary.drop(columns = 'geometry')
drought_summary.to_parquet("./Results/drought_intensity_magnitude_duration_summary.parquet", index=False)

#---------------------------------------------------------------------------------------------#
df_flood = df_clean[df_clean['is_flood']].copy()

# Flood excess (positive value)
df_flood['excess'] = df_flood['discharge'] - df_flood['p99']
# df_flood = df_flood[0:10000]

# Aggregate by station, scenario, season, time_period, and event ID
flood_summary = (
    df_flood.groupby(['station_name', 'scenario', 'time_period', 'season', 'flood_event_id'], observed = True)
    .agg(
        start_time=('time', 'min'),
        end_time=('time', 'max'),
        duration_weeks=('time', 'count'),
        magnitude=('excess', 'mean'),
    )
    .reset_index()
)

# Compute intensity
flood_summary['intensity'] = flood_summary['magnitude'] / flood_summary['duration_weeks']

# Ensure station_name and event_ids are integer
flood_summary['station_name'] = flood_summary['station_name'].astype(int)
flood_summary['flood_event_id'] = flood_summary['flood_event_id'].astype(int)

# Round magnitude and intensity
flood_summary['magnitude'] = flood_summary['magnitude'].round(2)
flood_summary['intensity'] = flood_summary['intensity'].round(2)


# Merge station metadata into flood summaries
flood_summary = flood_summary.merge(station_gdf_with_srex, on='station_name', how='left')
flood_summary = flood_summary.drop(columns = 'geometry')
flood_summary.to_parquet("./Results/flood_intensity_magnitude_duration_summary.parquet", index=False)

#---------------------------------------------------------------------------------------------#


