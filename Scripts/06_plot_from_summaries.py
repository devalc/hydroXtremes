#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:45:51 2025

@author: cdeval
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os

outdir = './Results/Figures/'

os.makedirs(outdir, exist_ok = True)

drought_summary = pd.read_parquet("./Results/drought_intensity_magnitude_duration_summary.parquet")
flood_summary = pd.read_parquet("./Results/flood_intensity_magnitude_duration_summary.parquet")


#---------------------------------------------------------------------------------------------#
#How will extreme high flows (floods) evolve under different RCP scenarios?

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

flood_summary_norm['duration_rel'] = flood_summary_norm['duration_weeks'] / flood_summary_norm['duration_median_hist']
flood_summary_norm['magnitude_rel'] = flood_summary_norm['magnitude'] / flood_summary_norm['magnitude_median_hist']
flood_summary_norm['intensity_rel'] = flood_summary_norm['intensity'] / flood_summary_norm['intensity_median_hist']

    
#---------------------------------------------------------------------------------------------#

def plot_flood_metric_by_time(df, save_path=None,  dpi =300):
    metrics = [
        ('duration_rel', 'Flood Duration (Relative to Historical Median)'),
        ('magnitude_rel', 'Flood Magnitude (Relative to Historical Median)'),
        ('intensity_rel', 'Flood Intensity (Relative to Historical Median)')
    ]
    
    # Clean and prep data
    df = df[(df['time_period'] != 'historical') & (df['scenario'] != 'hist')]
    df['time_period'] = pd.Categorical(df['time_period'], categories=['early', 'mid', 'end'], ordered=True)
    
    # Rename scenarios for better legend labels
    df['scenario'] = df['scenario'].replace({
        'rcp4p5': 'RCP4.5',
        'rcp8p5': 'RCP8.5'
    })

    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # For collecting legend items
    all_handles = []
    all_labels = []

    for i, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        # Create boxplot (legend is created here temporarily)
        sns.boxplot(
            data=df, x='time_period', y=metric, hue='scenario',
            ax=ax, showfliers=False, dodge=True
        )

        # Add baseline line
        line = ax.axhline(1, color='red', linestyle='--', label='Historical Baseline')

        # Grab handles/labels only once
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        # Add baseline to global legend if not already added
        if 'Historical Baseline' not in all_labels:
            all_handles.append(line)
            all_labels.append('Historical Baseline')

        # Now remove the subplot legend
        if ax.get_legend():
            ax.get_legend().remove()

        # Plot settings
        ax.set_yscale('log')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(['(a) Duration', '(b) Magnitude', '(c) Intensity'][i])
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # axes[-1].set_xlabel('Time Period')

    # Add the one shared global legend
    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        
    plt.show()
    


plot_flood_metric_by_time(flood_summary_norm, save_path = os.path.join(outdir, 'Figure1.png') )

#############################################################################


"""
Relative changes in flood characteristics (duration, magnitude, and intensity) across different time periods and climate scenarios (historical, RCP4.5, and RCP8.5). Each metric is normalized to the historical median value for each station to account for local variability. Boxplots represent distributions across all stations. The red dashed line marks the historical baseline (relative value = 1). Results show increasing trends in flood metrics under future scenarios, particularly under RCP8.5, suggesting stronger and more persistent high-flow events toward the end of the century.
"""

def plot_flood_metrics_by_season(df, save_path=None,  dpi =300):
    metrics = [
        ('duration_rel', '(a) Duration'),
        ('magnitude_rel', '(b) Magnitude'),
        ('intensity_rel', '(c) Intensity')
    ]

    # Clean data
    df = df[(df['time_period'] != 'historical') & (df['scenario'] != 'hist')]
    df['time_period'] = pd.Categorical(df['time_period'], categories=['early', 'mid', 'end'], ordered=True)

    # Define season order
    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
    
    # Rename scenarios for better legend labels
    df['scenario'] = df['scenario'].replace({
        'rcp4p5': 'RCP4.5',
        'rcp8p5': 'RCP8.5'
    })

    # Set up subplots: 3 rows (metrics) x 4 cols (seasons)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey='row')

    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, season in enumerate(season_order):
            ax = axes[row_idx, col_idx]
            season_df = df[df['season'] == season]

            sns.boxplot(
                data=season_df,
                x='time_period',
                y=metric,
                hue='scenario',
                ax=ax,
                showfliers=False,
                dodge=True
            )

            ax.axhline(1, color='red', linestyle='--', label='Historical Baseline')
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

            # Labeling
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('')

            if row_idx == 2:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('')

            if row_idx == 0:
                ax.set_title(f'{season}')

            # Remove subplot legends
            if ax.get_legend():
                ax.get_legend().remove()

    # One global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels), fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
    plt.show()

# Example usage:
plot_flood_metrics_by_season(flood_summary_norm, save_path = os.path.join(outdir, 'Figure2.png') )


#---------------------------------------------------------------------------------------------#

srex_gdf = gpd.read_file("./data/IPCC-WGI-reference-regions-v4_shapefile/IPCC-WGI-reference-regions-v4.shp")
srex_gdf = srex_gdf.to_crs('EPSG:4326')

# Step 1: Aggregate median relative flood metrics by SREX region and scenario
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

# Rename scenario labels
flood_region_summary['scenario'] = flood_region_summary['scenario'].replace({
    'rcp4p5': 'RCP4.5',
    'rcp8p5': 'RCP8.5'
})



# Step 2: Merge with SREX geometry
srex_with_flood_metrics = srex_gdf.merge(flood_region_summary, on=['Name'], how='left')


# Plot settings
vmin, vmax = 0.5, 2.5
cmap = 'inferno_r'
scenarios = flood_region_summary['scenario'].unique()

fig, axes = plt.subplots(3, len(scenarios), figsize=(16, 9),
                         gridspec_kw={'wspace': 0.0, 'hspace': 0.2})

row_labels = ['(a) Duration', '(b) Magnitude', '(c) Intensity']

# Plotting loop
for i, scenario in enumerate(scenarios):
    scenario_data = srex_with_flood_metrics[srex_with_flood_metrics['scenario'] == scenario]

    # Duration
    scenario_data.plot(
        column='rel_duration_median',
        cmap=cmap,
        ax=axes[0, i],
        legend=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black'
    )
    axes[0, i].set_title(scenario.upper())
    # axes[0, i].axis('off')
    
    # Add labels
    for _, row in scenario_data.iterrows():
        if row['geometry'] and not row['geometry'].is_empty:
            centroid = row['geometry'].centroid
            axes[0, i].annotate(
                text=row['Acronym'],
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=6,
                color='white'
                )

    # Magnitude
    scenario_data.plot(
        column='rel_magnitude_median',
        cmap=cmap,
        ax=axes[1, i],
        legend=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black'
    )
    # axes[1, i].axis('off')
    # Add labels
    for _, row in scenario_data.iterrows():
        if row['geometry'] and not row['geometry'].is_empty:
            centroid = row['geometry'].centroid
            axes[1, i].annotate(
                text=row['Acronym'],
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=6,
                color='white'
                )

    # Intensity
    scenario_data.plot(
        column='rel_intensity_median',
        cmap=cmap,
        ax=axes[2, i],
        legend=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black'
    )
    # axes[2, i].axis('off')
    # Add labels
    for _, row in scenario_data.iterrows():
        if row['geometry'] and not row['geometry'].is_empty:
            centroid = row['geometry'].centroid
            axes[2, i].annotate(
                text=row['Acronym'],
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=6,
                color='white'
                )

# Add rotated row labels
for j, label in enumerate(row_labels):
    axes[j, 0].text(
        -0.08, 0.5, label, va='center', ha='right',
        fontsize=12, rotation=90, transform=axes[j, 0].transAxes
    )

# Create a shared colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
sm.set_array([])  # Required to make the ScalarMappable work
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04)
cbar.set_label('', fontsize=16)

# plt.suptitle('Relative Flood Metrics by SREX Region and Scenario', fontsize=16)
plt.savefig(os.path.join(outdir, 'Figure3.png'), dpi = 300)
plt.show()



#---------------------------------------------------------------------------------------------#
# Create pivot tables
heatmap_duration = flood_region_summary.pivot(index='Name', columns='scenario', values='rel_duration_median')
heatmap_magnitude = flood_region_summary.pivot(index='Name', columns='scenario', values='rel_magnitude_median')
heatmap_intensity = flood_region_summary.pivot(index='Name', columns='scenario', values='rel_intensity_median')


# Optional: Set font scale for readability
sns.set(font_scale=0.9)

fig, axes = plt.subplots(3, 1, figsize=(14, 32), sharex=True)

# Heatmap 1 - Duration
sns.heatmap(
    heatmap_duration, annot=True, fmt=".2f", cmap='coolwarm', center=1,
    linewidths=0.5, ax=axes[0], cbar_kws={'label': 'Relative Duration'},
    yticklabels=True
)
axes[0].set_title('Flood Duration')
axes[0].set_ylabel('SREX Region')

# Heatmap 2 - Magnitude
sns.heatmap(
    heatmap_magnitude, annot=True, fmt=".2f", cmap='coolwarm', center=1,
    linewidths=0.5, ax=axes[1], cbar_kws={'label': 'Relative Magnitude'},
    yticklabels=True
)
axes[1].set_title('Flood Magnitude')
axes[1].set_ylabel('SREX Region')

# Heatmap 3 - Intensity
sns.heatmap(
    heatmap_intensity, annot=True, fmt=".2f", cmap='coolwarm', center=1,
    linewidths=0.5, ax=axes[2], cbar_kws={'label': 'Relative Intensity'},
    yticklabels=True
)
axes[2].set_title('Flood Intensity')
axes[2].set_ylabel('SREX Region')
axes[2].set_xlabel('Scenario')

# plt.suptitle('Supplementary Figure S1. Regional Median Flood Metrics Across Scenarios', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(os.path.join(outdir, 'Figure_S1.png'), dpi = 300)
plt.show()

#---------------------------------------------------------------------------------------------#

###DROUGHTS
#---------------------------------------------------------------------------------------------#

"""

Computing relative drought metrics normalized to the historical median per station
"""
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
drought_summary_norm['duration_rel'] = drought_summary_norm['duration_weeks'] / drought_summary_norm['duration_median_hist']
drought_summary_norm['magnitude_rel'] = drought_summary_norm['magnitude'] / drought_summary_norm['magnitude_median_hist']
drought_summary_norm['intensity_rel'] = drought_summary_norm['intensity'] / drought_summary_norm['intensity_median_hist']


#---------------------------------------------------------------------------------------------#
# Boxplots by scenario, resusing same flood function
#How do flood dynamics change over time and season?

def plot_drought_metric_by_time(df, save_path=None, dpi =300):
    metrics = [
        ('duration_rel', 'Drought Duration (Relative to Historical Median)'),
        ('magnitude_rel', 'Drought Magnitude (Relative to Historical Median)'),
        ('intensity_rel', 'Drought Intensity (Relative to Historical Median)')
    ]
    
    # Clean and prep data
    df = df[(df['time_period'] != 'historical') & (df['scenario'] != 'hist')]
    df['time_period'] = pd.Categorical(df['time_period'], categories=['early', 'mid', 'end'], ordered=True)
    
    # Rename scenarios for better legend labels
    df['scenario'] = df['scenario'].replace({
        'rcp4p5': 'RCP4.5',
        'rcp8p5': 'RCP8.5'
    })

    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # For collecting legend items
    all_handles = []
    all_labels = []

    for i, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        # Create boxplot (legend is created here temporarily)
        sns.boxplot(
            data=df, x='time_period', y=metric, hue='scenario',
            ax=ax, showfliers=False, dodge=True
        )

        # Add baseline line
        line = ax.axhline(1, color='red', linestyle='--', label='Historical Baseline')

        # Grab handles/labels only once
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        # Add baseline to global legend if not already added
        if 'Historical Baseline' not in all_labels:
            all_handles.append(line)
            all_labels.append('Historical Baseline')

        # Now remove the subplot legend
        if ax.get_legend():
            ax.get_legend().remove()

        # Plot settings
        ax.set_yscale('log')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(['(a) Duration', '(b) Magnitude', '(c) Intensity'][i])
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # axes[-1].set_xlabel('Time Period')

    # Add the one shared global legend
    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
    plt.show()


plot_drought_metric_by_time(drought_summary_norm, save_path = os.path.join(outdir, 'Figure4.png'))


#---------------------------------------------------------------------------------------------#
# Boxplots by season

def plot_drought_metrics_by_season(df, save_path=None, dpi =300):
    metrics = [
        ('duration_rel', '(a) Duration'),
        ('magnitude_rel', '(b) Magnitude'),
        ('intensity_rel', '(c) Intensity')
    ]

    # Clean data
    df = df[(df['time_period'] != 'historical') & (df['scenario'] != 'hist')]
    df['time_period'] = pd.Categorical(df['time_period'], categories=['early', 'mid', 'end'], ordered=True)

    # Define season order
    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
    
    # Rename scenarios for better legend labels
    df['scenario'] = df['scenario'].replace({
        'rcp4p5': 'RCP4.5',
        'rcp8p5': 'RCP8.5'
    })

    # Set up subplots: 3 rows (metrics) x 4 cols (seasons)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey='row')

    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, season in enumerate(season_order):
            ax = axes[row_idx, col_idx]
            season_df = df[df['season'] == season]

            sns.boxplot(
                data=season_df,
                x='time_period',
                y=metric,
                hue='scenario',
                ax=ax,
                showfliers=False,
                dodge=True
            )

            ax.axhline(1, color='red', linestyle='--', label='Historical Baseline')
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

            # Labeling
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('')

            if row_idx == 2:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('')

            if row_idx == 0:
                ax.set_title(f'{season}')

            # Remove subplot legends
            if ax.get_legend():
                ax.get_legend().remove()

    # One global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels), fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# Example usage:
plot_drought_metrics_by_season(drought_summary_norm, save_path = os.path.join(outdir, 'Figure5.png'))


#---------------------------------------------------------------------------------------------#
#chloropleth by region

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

# Rename scenario labels
drought_region_summary['scenario'] = drought_region_summary['scenario'].replace({
    'rcp4p5': 'RCP4.5',
    'rcp8p5': 'RCP8.5'
})


# Merge with SREX shapefile
srex_with_drought_metrics = srex_gdf.merge(drought_region_summary, on='Name', how='left')

# Plot settings
vmin, vmax = 0.0, 3.6
cmap = 'inferno_r'
scenarios = drought_region_summary['scenario'].unique()

fig, axes = plt.subplots(3, len(scenarios), figsize=(16, 9),
                         gridspec_kw={'wspace': 0.0, 'hspace': 0.2})

row_labels = ['(a) Duration', '(b) Magnitude', '(c) Intensity']

# Plotting loop
for i, scenario in enumerate(scenarios):
    scenario_data = srex_with_drought_metrics[srex_with_drought_metrics['scenario'] == scenario]


    # Duration
    scenario_data.plot(
        column='rel_duration_median',
        cmap=cmap,
        ax=axes[0, i],
        legend=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black'
    )
    axes[0, i].set_title(scenario.upper())
    # axes[0, i].axis('off')
    
    # Add labels
    for _, row in scenario_data.iterrows():
        if row['geometry'] and not row['geometry'].is_empty:
            centroid = row['geometry'].centroid
            axes[0, i].annotate(
                text=row['Acronym'],
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=6,
                color='white'
                )

    # Magnitude
    scenario_data.plot(
        column='rel_magnitude_median',
        cmap=cmap,
        ax=axes[1, i],
        legend=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black'
    )
    # axes[1, i].axis('off')
    # Add labels
    for _, row in scenario_data.iterrows():
        if row['geometry'] and not row['geometry'].is_empty:
            centroid = row['geometry'].centroid
            axes[1, i].annotate(
                text=row['Acronym'],
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=6,
                color='white'
                )

    # Intensity
    scenario_data.plot(
        column='rel_intensity_median',
        cmap=cmap,
        ax=axes[2, i],
        legend=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black'
    )
    # axes[2, i].axis('off')
    # Add labels
    for _, row in scenario_data.iterrows():
        if row['geometry'] and not row['geometry'].is_empty:
            centroid = row['geometry'].centroid
            axes[2, i].annotate(
                text=row['Acronym'],
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=6,
                color='white'
                )

# Add rotated row labels
for j, label in enumerate(row_labels):
    axes[j, 0].text(
        -0.08, 0.5, label, va='center', ha='right',
        fontsize=12, rotation=90, transform=axes[j, 0].transAxes
    )

# Create a shared colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
sm.set_array([])  # Required to make the ScalarMappable work
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04)
cbar.set_label('', fontsize=12)

# plt.suptitle('Relative Drought Metrics by SREX Region and Scenario', fontsize=16)
plt.savefig(os.path.join(outdir, 'Figure6.png'), dpi = 300)
plt.show()


#---------------------------------------------------------------------------------------------#
#heatmap

# Create pivot tables
heatmap_duration_dr = drought_region_summary.pivot(index='Name', columns='scenario', values='rel_duration_median')
heatmap_magnitude_dr = drought_region_summary.pivot(index='Name', columns='scenario', values='rel_magnitude_median')
heatmap_intensity_dr = drought_region_summary.pivot(index='Name', columns='scenario', values='rel_intensity_median')


# Optional: Set font scale for readability
sns.set(font_scale=0.9)

fig, axes = plt.subplots(3, 1, figsize=(14, 32), sharex=True)

# Heatmap 1 - Duration
sns.heatmap(
    heatmap_duration_dr, annot=True, fmt=".2f", cmap='coolwarm', center=1,
    linewidths=0.5, ax=axes[0], cbar_kws={'label': 'Relative Duration'},
    yticklabels=True
)
axes[0].set_title('Drought Duration')
axes[0].set_ylabel('SREX Region')

# Heatmap 2 - Magnitude
sns.heatmap(
    heatmap_magnitude_dr, annot=True, fmt=".2f", cmap='coolwarm', center=1,
    linewidths=0.5, ax=axes[1], cbar_kws={'label': 'Relative Magnitude'},
    yticklabels=True
)
axes[1].set_title('Drought Magnitude')
axes[1].set_ylabel('SREX Region')

# Heatmap 3 - Intensity
sns.heatmap(
    heatmap_intensity_dr, annot=True, fmt=".2f", cmap='coolwarm', center=1,
    linewidths=0.5, ax=axes[2], cbar_kws={'label': 'Relative Intensity'},
    yticklabels=True
)
axes[2].set_title('Drought Intensity')
axes[2].set_ylabel('SREX Region')
axes[2].set_xlabel('Scenario')

# plt.suptitle('Supplementary Figure S2. Regional Median Drought Metrics Across Scenarios', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(outdir, 'Figure_S2.png'), dpi = 300)
plt.show()
