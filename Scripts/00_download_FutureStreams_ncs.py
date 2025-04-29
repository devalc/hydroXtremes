# -*- coding: utf-8 -*-
"""
Updated on Saturday Mar 10 11:53:44 2024

Purpose: Download FutureStreams netcdf files from
 https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/discharge/

@author: devalc
"""

import os
import time
import subprocess
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests

# Specify the download location path
download_location = 'F:/FutStreams_data/data/'  

#base urls
base_url = 'https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/discharge/'
scenarios = ['hist', 'rcp4p5', 'rcp8p5']

# Record the start time
start_time = time.time()

def get_netcdf_paths_for_model(base_url, scenario, model):
    model_folder_url = urljoin(base_url, f"{scenario}/{model}/")
    response = requests.get(model_folder_url)

    if response.status_code == 404:
        print(f"Model folder not found: {model_folder_url}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    file_list = [a['href'] for tr in soup.find_all('tr', class_='object data-object extension-nc') for a in tr.find_all('a', href=True)]

    netcdf_paths = [urljoin(model_folder_url, file_name) for file_name in file_list]
    return netcdf_paths

all_netcdf_paths = []

for scenario in scenarios:
    if scenario == 'hist':
        models = ['E2O', 'gfdl', 'hadgem', 'ipsl', 'miroc', 'noresm']
    else:
        models = ['gfdl', 'hadgem', 'ipsl', 'miroc', 'noresm']

    for model in models:
        model_paths = get_netcdf_paths_for_model(base_url, scenario, model)
        all_netcdf_paths.extend(model_paths)

# Download files using wget with 10 parallel downloads
for i in range(0, len(all_netcdf_paths), 10):
    paths_chunk = all_netcdf_paths[i:i + 10]

    # Start subprocesses for parallel downloads
    processes = []
    for path in paths_chunk:
        command = ['wget', '-q', '-nc', '--no-check-certificate', '--show-progress', '-P', download_location, path]
        process = subprocess.Popen(command)
        processes.append(process)

    # Wait for all subprocesses to finish
    for process in processes:
        process.wait()

print("Download completed.")

# Record the end time
end_time = time.time()

# Calculate the total time taken in hours
total_time_hours = (end_time - start_time) / 3600

# Print the total time in hours
print(f"Total execution time: {total_time_hours:.2f} hours")