# -*- coding: utf-8 -*-
"""
Updated on Friday Mar 15 05:53:44 2024

Purpose: Plot the extracted parquet file for testing
@author: devalc
"""
import pandas as pd
import matplotlib.pyplot as plt
import random

# Specify the path to the extracted Parquet file
parquet_file = "/Volumes/Personal/streamflow/merged_all_models_and_all_scenarios.parquet"

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file)

# Get unique station names
station_names = df['station_name'].unique()

# Randomly select 100 station names
random_stations = random.sample(list(station_names), 100)

# Plot time series for randomly selected stations
plt.figure(figsize=(10, 6))
for station_name in random_stations:
    # Extract data for the station
    station_data = df[df['station_name'] == station_name]
    timeseries = station_data['time']
    discharge = station_data['discharge']
    
    # Plot
    plt.plot(timeseries, discharge, label=f"Station {station_name}", alpha=0.7)

# Add legend
plt.legend()

# Add labels and title
plt.title("Time Series for 100 Randomly Selected Stations")
plt.xlabel("Time")
plt.ylabel("Discharge")

# Show grid
plt.grid(True)

# Show plot
plt.show()
