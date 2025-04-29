# -*- coding: utf-8 -*-
"""
Updated on Friday Mar 15 04:53:44 2024

Purpose: Process FutureStreams netcdf files to extract timeseries at GRDC station locations
         FutureStreams was downloaded form: https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/discharge/
         GRDC data was downloaded from: https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/StationCatalogue
@author: devalc
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import time

# Function to process a single NetCDF file
def process_netcdf(nc_file, grdc_data):
    # Load the NetCDF file
    print(f"Processing file: {nc_file}")
    ds = xr.open_dataset(nc_file)

    # Create an empty list to store data for each station
    data = []

    # Loop over GRDC stations and extract values
    for idx, row in grdc_data.iterrows():
        station_name = row['grdc_no']
        lat = row['lat']
        lon = row['long']
        
        # Find the nearest grid point in the NetCDF dataset
        lat_idx = np.abs(ds.latitude - lat).argmin().item()
        lon_idx = np.abs(ds.longitude - lon).argmin().item()
        
        # Extract the time series data of "discharge"
        timeseries = ds['discharge'][:, lat_idx, lon_idx].values
        
        # Extract time values
        time_values = ds['time'].values
        
        # Create DataFrame for the station
        df = pd.DataFrame({'time': time_values, 'discharge': timeseries})
        
        # Add station name as a column
        df['station_name'] = station_name
        
        # Append DataFrame to the list
        data.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data)
    
    # Print the head of the extracted data
    #print(f"Head of extracted data for file {nc_file}:")
    #print(combined_df.head())

    return combined_df

# Start timing
start_time = time.time()

# Load the Excel file and extract relevant columns
excel_file = r"F:\FutStreams_data\grdc_stations\GRDC_Stations.xlsx"
grdc_data = pd.read_excel(excel_file, usecols=["grdc_no", "long", "lat"])

# Select first 10 stations for testing
grdc_data = grdc_data

# Directory containing NetCDF files
nc_files_directory = "F:/FutStreams_data/data/"

# Define output directory
output_directory = "F:/FutStreams_data/data/extracted_parquet_files/"
os.makedirs(output_directory, exist_ok=True)

# Loop over NetCDF files in the directory
for nc_file in os.listdir(nc_files_directory):
    #print(f"File in loop: {nc_file}")
    if nc_file.endswith(".nc"):
        nc_file_path = os.path.join(nc_files_directory, nc_file)
        
        # Process the NetCDF file
        combined_df = process_netcdf(nc_file_path, grdc_data)
        
        # Specify the output Parquet file path
        base_name = os.path.splitext(os.path.basename(nc_file))[0]
        output_parquet_file = os.path.join(output_directory, f"{base_name}.parquet")
        
        # Write the DataFrame to Parquet
        combined_df.to_parquet(output_parquet_file, index=False)
        
        # Specify the output Excel file path
        output_excel_file = os.path.join(output_directory, f"{base_name}.xlsx")

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
elapsed_hours = elapsed_time / 3600
print(f"Total time taken: {elapsed_hours} hours")
print("Extraction and saving to Parquet completed successfully.")
