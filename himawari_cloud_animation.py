__version__ = "2025-07-31"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

"""
Himawari-8 Cloud Type Animation, based on Sam Green's example:
https://21centuryweather.github.io/21st-Century-Weather-Software-Wiki/datasets/himawari-ahi.html

Change ## USER INPUT ## section as requried (dates, domain, user etc)

mp4 quality is set to 30, with lower values being better quality but higher file size

Please cite/acknowledge the following people if this data has been used for publications/presentations: 
Samuel Green - ORCID 0000-0003-1129-4676; Mat Lipson - ORCID 0000-0001-5322-1796; Kimberley Reid - ORCID 0000-0001-5972-6015.

Run with:
    module use /g/data/xp65/public/modules; module load conda/analysis3
    python himawari_cloud_animation.py
"""

import xarray as xr
import numpy as np
import re
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.ticker as mticker

from dask.distributed import Client

####################################################################
## USER INPUT ##

# project and user for output directory
project = 'fy29'
user = 'mjl561'

# subset domain of interest
xmin, xmax, ymin, ymax = 140.95, 154.30, -39.35, -24.54  # ECL subdomain
# xmin, xmax, ymin, ymax = 110, 155, -45, -9               # Australian domain
xmin, xmax, ymin, ymax = 145, 165, -35, -15     # SE Queensland domain

# dates of interest
sdate = '2025-03-06 00:00'
edate = '2025-03-08 00:00'

# output directory
output_dir = f'/scratch/{project}/{user}/himawari_cloud_animation/'

####################################################################

def main(xmin, xmax, ymin, ymax, sdate, edate):
    '''
    Main function to create an animation of Himawari-8 cloud types.
    '''

    ds_z = xr.open_zarr("/g/data/su28/himawari-ahi/cloud/ct/aus_regional_domain/S_NWC_CT_HIMA08_HIMA-N-NR.zarr", consolidated=True)

    # Select the time range
    ds = ds_z.sel(time=slice(sdate, edate))
    # Select the spatial domain
    ds = ds.sel(lat=slice(ymax, ymin), lon=slice(xmin, xmax))

    # choose which cloud types to include in the animation (per get_categories function)
    included_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Create discrete colormap and normalization
    cmap, norm = get_cmap()
    
    # get categories for cloud types
    category_dict = get_categories(included_values)

    for time in ds.time.values:
        print(f"Processing time: {time}")

        # Create animation frame for each time step
        da = ds.sel(time=time).ct
        # if no data in da, then skip
        create_animation_frame(da, cmap, norm, category_dict) if da.notnull().all() else print(f"skipping.")

    # now create animation from the created frames
    fnamein = f'{output_dir}/himawari_cloud_type_*.png'
    fnameout = f'{output_dir}/himawari_cloud_type_animation.mp4'
    make_mp4(fnamein,fnameout,fps=24,quality=30)

    return ds_z


def get_categories(values):

    comment = '''
        1:  Cloud-free land; 
        2:  Cloud-free sea; 
        3:  Snow over land;  
        4:  Sea ice; 
        5:  Very low clouds; 
        6:  Low clouds; 
        7:  Mid-level clouds;  
        8:  High opaque clouds; 
        9:  Very high opaque clouds;  
        10:  Fractional clouds; 
        11:  High semitransparent thin clouds;  
        12:  High semitransparent moderately thick clouds;  
        13:  High semitransparent thick clouds;  
        14:  High semitransparent above low or medium clouds;  
        15:  High semitransparent above snow/ice'''

    # Use regex to extract values and labels
    matches = re.findall(r'(\d+):\s+([^;]+)', comment)

    # Convert to dictionary
    category_dict = {int(num): desc.strip() for num, desc in matches}

    # Filter to only include categories that match the specified values
    filtered_categories = {k: v for k, v in category_dict.items() if k in values}

    # Print result for specified values only
    print("Categories:")
    for k, v in sorted(filtered_categories.items()):
        print(f"{k}: {v}")

    return filtered_categories

def create_animation_frame(da, cmap, norm, category_dict):

    # Create a figure
    plt.close('all')  # Close any existing figures
    fig = plt.figure(figsize=(16, 8))

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=130))
    ax.coastlines(resolution="50m", color='white')

    # Filter data to only include values that are in category_dict
    mask = da.isin(list(category_dict.keys()))
    ct_filtered = da.where(mask)

    img = ct_filtered.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)
    
    # Add a discrete colorbar
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("Cloud Type (CT)", fontsize=12)
    cbar.set_ticks(list(category_dict.keys()))  # Only show ticks for filtered categories
    cbar.set_ticklabels([f"{k}: {v}" for k, v in sorted(category_dict.items())])  # Custom labels for filtered categories only

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=1.0)
    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels

    # Increase the number of ticks on the x-axis
    gl.xlocator = mticker.FixedLocator(np.arange(70, 190, 20))  # Adjust step size as needed

    # create string for the time
    time_str = str(da.time.values).replace('T', ' ').replace('Z', '')[:16]
    time_str_fname = time_str.replace(' ', '_').replace(':', '')
    plt.title(f"Himawari Cloud Type at {time_str}", fontsize=14)
    # plt.show()
    plt.savefig(f"{output_dir}/himawari_cloud_type_{time_str_fname}.png", bbox_inches='tight', dpi=150)

def get_cmap():
    """
    Create a discrete colormap for cloud type visualization based on tab20 colormap.
    Returns a ListedColormap suitable for cloud type categories 1-14.
    """

    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Set the min and max colors from tab20
    min_color = mpl.colormaps['tab20'](0 / 20)
    max_color = mpl.colormaps['tab20'](19 / 20)  # Last color of tab20

    # Generate intermediate colors by linearly spacing them within tab20
    num_classes = len(values)
    colors = [mpl.colormaps['tab20'](i / (num_classes - 1)) for i in range(num_classes)]

    # Ensure min/max colors are set explicitly
    colors[0] = min_color
    colors[-1] = max_color

    # Create discrete colormap
    cmap = mcolors.ListedColormap(colors)

    # Define boundaries for normalization (each value gets its own bin)
    boundaries = np.arange(min(values) - 0.5, max(values) + 1.5, 1)  # Adjusted for correct binning
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    return cmap, norm

def sams_full_example():

    """Sam Green's example for cloud type visualization:
    https://21centuryweather.github.io/21st-Century-Weather-Software-Wiki/datasets/himawari-ahi.html
    This function visualizes cloud types from Himawari-8 data using a discrete colormap.
    """

    ds_z = xr.open_zarr("/g/data/su28/himawari-ahi/cloud/ct/aus_regional_domain/S_NWC_CT_HIMA08_HIMA-N-NR.zarr", consolidated=True)

    comment = '1:  Cloud-free land; 2:  Cloud-free sea; 3:  Snow over land;  4:  Sea ice; 5:  Very low clouds; 6:  Low clouds; 7:  Mid-level clouds;  8:  High opaque clouds; 9:  Very high opaque clouds;  10:  Fractional clouds; 11:  High semitransparent thin clouds;  12:  High semitransparent moderately thick clouds;  13:  High semitransparent thick clouds;  14:  High semitransparent above low or medium clouds;  15:  High semitransparent above snow/ice'

    # Use regex to extract values and labels
    matches = re.findall(r'(\d+):\s+([^;]+)', comment)

    # Convert to dictionary
    category_dict = {int(num): desc.strip() for num, desc in matches}

    # Print result
    for k, v in sorted(category_dict.items())[:14]:
        print(f"{k}: {v}")

    fig = plt.figure(figsize=(16, 8))

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=130))
    ax.coastlines(resolution="50m", color='white')

    # Define the values and corresponding colors from tab20
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Adjust based on your dataset
    # Set the min and max colors from tab20
    min_color = mpl.colormaps['tab20'](0 / 20)
    max_color = mpl.colormaps['tab20'](19 / 20)  # Last color of tab20

    # Generate intermediate colors by linearly spacing them within tab20
    num_classes = len(values)
    colors = [mpl.colormaps['tab20'](i / (num_classes - 1)) for i in range(num_classes)]

    # Ensure min/max colors are set explicitly
    colors[0] = min_color
    colors[-1] = max_color

    # Create discrete colormap
    cmap = mcolors.ListedColormap(colors)
    # Define boundaries for normalization (each value gets its own bin)
    boundaries = np.arange(min(values) - 0.5, max(values) + 1.5, 1)  # Adjusted for correct binning
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    img = ds_z.sel(time='2022-01-01T03:00:00.000000000').ct.plot(ax=ax, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)

    # Add a discrete colorbar
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("Cloud Type (CT)", fontsize=12)
    cbar.set_ticks(values)  # Ensure ticks match category values
    cbar.set_ticklabels([f"{k}: {v}" for k, v in sorted(category_dict.items())[:14]])  # Custom labels

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=1.0)
    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels

    # Increase the number of ticks on the x-axis
    gl.xlocator = mticker.FixedLocator(np.arange(70, 190, 20))  # Adjust step size as needed
    plt.show()


def make_mp4(fnamein,fnameout,fps=9,quality=26):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatability across OS.
        fnamein (string): The image files to create animation from, with glob wildcards (*).
        fnameout (string): The output filename
        fps (float): The frames per second. Default 6.
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

    # resize output to blocksize for maximum capatability between different OS
    macro_block_size=16
    if out_h % macro_block_size > 0:
        out_h += macro_block_size - (out_h % macro_block_size)
    if out_w % macro_block_size > 0:
        out_w += macro_block_size - (out_w % macro_block_size)

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 1 and 51 inclusive"

    # use ffmpeg command to create mp4
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{fnamein}" \
        -vcodec libx264 -crf {quality} -s {out_w}x{out_h} -pix_fmt yuv420p -y {fnameout}'
    os.system(command)

    return f'completed, see: {fnameout}'

if __name__ == "__main__":

    client = Client()
    client

    # check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds_z = main(xmin, xmax, ymin, ymax, sdate, edate)