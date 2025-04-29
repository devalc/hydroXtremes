# -*- coding: utf-8 -*-
"""
Updated on Monday Mar 25 22:01:13 2024

Purpose: Test the merge parquet file was created as expeted. 
Code plots all available scenarios and models for a randomly selected station.
@author: devalc
"""

import pandas as pd
import matplotlib.pyplot as plt
import random

# Read the merged Parquet file
merged_df = pd.read_parquet('./merged_all_models_and_all_scenarios.parquet')

# Get a random station name
random_station = random.choice(merged_df['station_name'].unique())

# Filter DataFrame for the random station
station_df = merged_df[merged_df['station_name'] == random_station]

# Get a random numeric column
numeric_columns = station_df.select_dtypes(include=['number']).columns
random_column = random.choice(numeric_columns)

# Get unique combinations of model and scenario for the random station
unique_combinations = station_df[['model', 'scenario']].drop_duplicates()

# Plot all available models and scenarios for the random station and column
plt.figure(figsize=(10, 6))
for index, row in unique_combinations.iterrows():
    model = row['model']
    scenario = row['scenario']
    
    # Filter DataFrame for the current model and scenario
    subset_df = station_df[(station_df['model'] == model) & (station_df['scenario'] == scenario)]
    
    # Plot the values
    plt.plot(subset_df['time'], subset_df[random_column], label=f"Model: {model}, Scenario: {scenario}")

plt.title(f"Station: {random_station}, Column: {random_column}")
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
