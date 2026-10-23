"""
This script produces a 2D plots of the orography for different model resolutions
for the new higher resolution ACCESS-AM3 model.

author: Mat Lipson m.lipson@unsw.edu.au
institution: 21st Century Weather, UNSW Sydney

Gadi environment: module use /g/data/xp65/public/modules; module load conda/analysis3-25.08
"""


import os
import xarray as xr
import rioxarray as rxr
import iris
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase

ancildir = '/g/data/gb02/public/AM3/ancils/u-dj813_om3-025deg'
plotpath = '/home/561/mjl561/postdoc/02-Projects/P76-ECOMIP'
reses = ['n96e','n512e','n2560e']

# domain around Australia
xmin, xmax, ymin, ymax = 110, 160, -50, 0
# domain around Australia with NZ
xmin, xmax, ymin, ymax = 100, 180, -50, 0

def main():

    # if plotpath does not exist, create it

    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    for res in reses:

        orogpath = f'{ancildir}/{res}/mom025_20250315/orography/gmted_ramp2/qrparm.orog_mean'
        lsmpath = f'{ancildir}/{res}/mom025_20250315/land_sea_mask/etop01/qrparm.landfrac'


        with open(orogpath): pass
        print(f'loading {orogpath}')
        cb = iris.load_cube(orogpath,constraint='surface_altitude')     

        with open(lsmpath): pass
        print(f'loading {lsmpath}')
        lsm = iris.load_cube(lsmpath)

        ds = xr.DataArray.from_iris(cb)
        # mask out ocean
        ds = ds.where(lsm.data>0.5)


        au = ds.sel(longitude=slice(xmin,xmax),latitude=slice(ymin,ymax))
        
        # plot

        res_km = int(au.rio.resolution()[1] * 111.32)

        print(f'resolution: {res} ~{res_km} km')
    
        proj = ccrs.PlateCarree()
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,8),subplot_kw={'projection': proj})
        # plot with colorbar on bottom
        im = au.plot(ax=ax, cmap='terrain',vmin=0,vmax=2500, add_colorbar=False)
        ax.set_title(f'{res} orography', fontsize=14)

        # colorbar
        cbar = custom_cbar(ax,im,cbar_loc='bottom')
        cbar.ax.set_xlabel('orography [m]')
        cbar.ax.tick_params(labelsize=8)
        
        # # for cartopy
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(color='0.5',linewidth=0.5,zorder=5)
        ax.set_extent([xmin,xmax,ymin,ymax],crs=proj)
        ax = distance_bar(ax,distance=res_km)

        # print filename on plot
        ax.text(0.5, 0.01, f'orography: {orogpath}', fontsize=5, ha='center', va='bottom', transform=ax.transAxes)
        ax.text(0.5, 0.02, f'land-sea mask: {lsmpath}', fontsize=5, ha='center', va='bottom', transform=ax.transAxes)
    
        plt.savefig(f'{plotpath}/orography_{res}.png',dpi=200, bbox_inches='tight')

def distance_bar(ax,distance=100):
    """
    Add a distance bar to the plot with geodesic distance in km
    """

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    xdist = abs(xlims[1]-xlims[0])
    offset = 0.03*xdist

    # plot distance bar
    start = (xlims[0]+offset,ylims[0]+offset)
    end = cgeo.Geodesic().direct(points=start,azimuths=90,distances=distance*1000).flatten()
    ax.plot([start[0],end[0]],[start[1],end[1]], color='0.65', linewidth=1.5)
    ax.text(start[0]+offset/7,start[1]+offset/5, f'{distance} km', 
        fontsize=9, ha='left',va='bottom', color='0.65')

    return ax

def custom_cbar(ax,im,cbar_loc='right',ticks=None):
    """
    Create a custom colorbar
    """

    if cbar_loc == 'right':
        cax = inset_axes(ax,
            width='4%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, ticks = ticks)

    elif cbar_loc == 'far_right':
        cax = inset_axes(ax,
            width='4%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.25, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, ticks = ticks)
    
    else:
        cbar_loc == 'bottom'
        cax = inset_axes(ax,
            width='100%',  # % of parent_bbox width
            height='4%',
            loc='lower left',
            bbox_to_anchor=(0, -0.15, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, orientation='horizontal', ticks = ticks)

    return cbar

if __name__ == '__main__':
    main()
