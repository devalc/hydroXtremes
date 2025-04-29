# -*- coding: utf-8 -*-
"""
Updated on Monday Mar 25 21:51:23 2024

Purpose: merge parquet files for different models and scenarios into one
@author: devalc
"""

import os
import pandas as pd

# Directory where Parquet files are located
directory = "/Volumes/Personal/streamflow/extracted_parquet_files"

# Initialize an empty list 
dfs = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.parquet'):
        # Extract model and scenario from the filename
        model = filename.split('_')[3]
        scenario = filename.split('_')[4]
        
        # Read Parquet file into DataFrame
        df = pd.read_parquet(os.path.join(directory, filename))
        
        # Add model and scenario columns
        df['model'] = model
        df['scenario'] = scenario
        
        # Append DataFrame to the list
        dfs.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(dfs, ignore_index=True)

# Save merged DataFrame to a new Parquet file
merged_df.to_parquet('merged_all_models_and_all_scenarios.parquet', index=False)
