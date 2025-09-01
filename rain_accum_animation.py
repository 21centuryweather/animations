#!/usr/bin/env python3
"""
Plot precipitation accumulation for a single experiment

Script to plot instantaneous and cumulative precipitation 
overlaid on a single panel for one specific experiment.

Author: Mathew Lipson <m.lipson@unsw.edu.au>
Date: 2025-08-30
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
import glob
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##############################################################################
# Configuration - EDIT THESE VALUES
##############################################################################

# Specific configuration
DOMAIN = 'GAL9'  # or 'RAL3P2'
EXPERIMENT = 'CCIv2_GAL9'  # specific experiment name
VARIABLE = 'total_precipitation_rate'  # or 'stratiform_rainfall_flux' for RAL3

# Paths
cylc_id = 'rns_ostia_NA_2020'
datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
plotpath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/figures'

##############################################################################
# Main Functions
##############################################################################

def main():
    """Main function"""
    print(f"Plotting {EXPERIMENT} precipitation...")
    
    # Create output directory
    os.makedirs(plotpath, exist_ok=True)
    
    # Load data
    ds = load_data().compute()
    
    print(f"Data shape: {ds.shape}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Pre-calculate cumulative sum for all time steps
    print("Pre-calculating cumulative sums...")
    ds_cumsum = ds.cumsum(dim='time').compute()
    
    # Plot all frames (or set n_frames to a smaller number for testing)
    n_frames = ds.time.size
    # n_frames = 20  # Uncomment this line for testing with fewer frames
    
    print(f"Creating {n_frames} frame plots...")
    for i in range(n_frames):
        print(f"Plotting time index {i}...")
        plot_single_frame(ds, i, ds_cumsum)

        # Garbage collection every 100 frames to manage memory
        if (i + 1) % 100 == 0:
            gc.collect()
    
    # Create MP4 animation from the frame files
    print("Creating MP4 animation...")
    mp4_qual = 32
    frame_pattern = f'{plotpath}/{VARIABLE}_single_{DOMAIN}_t*.png'
    mp4_output = f'{plotpath}/{VARIABLE}_single_{DOMAIN}_animation_q{mp4_qual}'
    result = make_mp4(frame_pattern, mp4_output, fps=48, quality=mp4_qual)
    
    # Delete PNG files that were used to make the movie
    print("Cleaning up PNG files...")
    png_files = sorted(glob.glob(frame_pattern))
    for png_file in png_files:
        if os.path.exists(png_file):
            os.remove(png_file)
            print(f"Deleted {png_file}")
    
    print(f"Completed! Created animation: {mp4_output}.mp4")

def load_data():
    """Load the precipitation data"""
    filename = f'{datapath}/{VARIABLE}/{EXPERIMENT}_{VARIABLE}.nc'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    print(f"Loading {filename}")
    ds = xr.open_dataset(filename)
    
    # Get the data variable and convert to mm/hour
    data_vars = [var for var in ds.data_vars if var not in ['time_bnds']]
    data = ds[data_vars[0]] * 3600  # Convert from kg m-2 s-1 to mm/hour
    
    return data

def plot_single_frame(ds, time_index, ds_cumsum):
    """Plot a single time frame with overlaid instantaneous and cumulative precipitation"""
    
    # Get time information
    time_str = str(ds.time.values[time_index])[:19].replace('T', ' ')
    
    # Select data for this time step
    instant_data = ds.isel(time=time_index)
    cumsum_data = ds_cumsum.isel(time=time_index)
    
    # Mask zero values
    instant_data = instant_data.where(instant_data > 0)
    cumsum_data = cumsum_data.where(cumsum_data > 0)
    
    # Set up colorbar ranges
    instant_vmax = 20  # mm/hour
    cumsum_max = ds_cumsum.isel(time=-1).max().values
    cumsum_vmax = 1500  # mm (hardcoded for consistency)
    
    # Define contour levels
    instant_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                     0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cumsum_levels = np.linspace(0, cumsum_vmax, 21)
    
    # Define custom tick locations (every 2nd level)
    instant_ticks = instant_levels[::2]  # Every 2nd level: [0, 0.1, 0.2, 0.3, ...]
    cumsum_ticks = cumsum_levels[::2]    # Every 2nd level: [0, 150, 300, ...]
    
    # Create the plot
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})
    
    # Plot instantaneous precipitation (Blues) - background
    im1 = instant_data.plot(ax=ax, cmap='Blues', vmin=0, vmax=instant_vmax,
                           levels=instant_levels, extend='max', add_colorbar=False,
                           transform=proj, alpha=1.0)
    
    # Plot cumulative precipitation (Purples) - overlay
    im2 = cumsum_data.plot(ax=ax, cmap='Purples', vmin=0, vmax=cumsum_vmax,
                          levels=cumsum_levels, extend='max', add_colorbar=False,
                          transform=proj, alpha=0.8)
    
    # Set title
    ax.set_title(f'ACCESS-rAM3: {EXPERIMENT}\n{time_str}', fontsize=12)
    
    # Add coastlines and set extent
    ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
    left, bottom, right, top = get_bounds_for_cartopy(ds)
    ax.set_extent([left, right, bottom, top], crs=proj)
    
    # Add latitude/longitude labels and ticks
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    
    # Add colorbars
    cbar1 = custom_cbar(ax, im1, cbar_loc='bottom', ticks=instant_ticks)
    cbar1.ax.set_xlabel('instantaneous precipitation [mm/hour]', fontsize=8)
    cbar1.ax.tick_params(labelsize=7)
    
    cbar2 = custom_cbar(ax, im2, cbar_loc='right', ticks=cumsum_ticks)
    cbar2.ax.set_ylabel('cumulative precipitation [mm]', fontsize=8, rotation=90, labelpad=3)
    cbar2.ax.tick_params(labelsize=7)
    
    # Save figure
    fname = f'{plotpath}/{VARIABLE}_single_{DOMAIN}_t{time_index:05d}.png'
    print(f'Saving figure to {fname}')
    fig.savefig(fname, dpi=200, bbox_inches='tight')

    # Explicit memory cleanup
    plt.close(fig)
    plt.clf()
    plt.cla()
    
    return fname

##############################################################################
# Helper Functions
##############################################################################

def make_mp4(fnamein, fnameout, fps=24, quality=23):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatibility across OS.
        fnamein (string): The image files to create animation from, with wildcards (*).
        fnameout (string): The output filename (excluding extension)
        fps (float): The frames per second.
        quality (float): quality ranges 0 to 51, 51 being worst.
    '''

    import glob
    import imageio.v2 as imageio

    # collect animation frames
    fnames = sorted(glob.glob(fnamein))
    if len(fnames)==0:
        print('no files found to process, check fnamein')
        return
    img_shp = imageio.imread(fnames[0]).shape
    out_h, out_w = img_shp[0],img_shp[1]

    # resize output to blocksize for maximum compatibility between different OS
    macro_block_size=16
    if out_h % macro_block_size > 0:
        out_h += macro_block_size - (out_h % macro_block_size)
    if out_w % macro_block_size > 0:
        out_w += macro_block_size - (out_w % macro_block_size)

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 0 and 51 inclusive"

    # use ffmpeg command to create mp4
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{fnamein}" \
        -vcodec libx264 -crf {quality} -s {out_w}x{out_h} -pix_fmt yuv420p -y {fnameout}.mp4'
    os.system(command)

    return f'completed, see: {fnameout}.mp4'

def get_bounds_for_cartopy(ds, y_dim='latitude', x_dim='longitude'):
    """Get geographic bounds from dataset"""

    left = float(ds[x_dim].min())
    right = float(ds[x_dim].max())
    top = float(ds[y_dim].max())
    bottom = float(ds[y_dim].min())

    resolution_y = (top - bottom) / (ds[y_dim].size - 1)
    resolution_x = (right - left) / (ds[x_dim].size - 1)

    top = round(top + resolution_y/2, 6)
    bottom = round(bottom - resolution_y/2, 6)
    right = round(right + resolution_x/2, 6)
    left = round(left - resolution_x/2, 6)

    if resolution_y < 0:
        top, bottom = bottom, top
    if resolution_x < 0:
        left, right = right, left

    return left, bottom, right, top

def custom_cbar(ax, im, cbar_loc='right', ticks=None):
    """Create a custom colorbar"""
    import matplotlib.ticker as mticker

    if cbar_loc == 'right':
        cax = inset_axes(ax,
            width='4%',
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm)
        cbar.ax.yaxis.set_label_position('left')
    else:
        # bottom
        cax = inset_axes(ax,
            width='100%',
            height='4%',
            loc='lower left',
            bbox_to_anchor=(0, -0.15, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm, orientation='horizontal')

    # Set custom ticks if provided
    if ticks is not None:
        cbar.set_ticks(ticks)

    cbar.formatter = mticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-6, 6))
    cbar.update_ticks()

    return cbar

##############################################################################

if __name__ == "__main__":
    main()
