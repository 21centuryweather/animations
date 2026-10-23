"""
This script produces a 3D perspective view of the orography for different model resolutions
for the new higher resolution ACCESS-AM3 model. The plot has a simple vertical exaggeration 
to visualize the terrain. 

author: Mat Lipson m.lipson@unsw.edu.au
institution: 21st Century Weather, UNSW Sydney

Gadi environment: module use /g/data/xp65/public/modules; module load conda/analysis3-25.08
"""

import os
import numpy as np
import xarray as xr
import iris
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

home = os.getenv('HOME')
user = os.getenv('USER')

plotpath = f'/scratch/gb02/{user}/3d_orography'
ancildir = '/g/data/gb02/public/AM3/ancils/u-dj813_om3-025deg'
reses = ['n48e', 'n96e', 'n512e', 'n2560e']

######## Domains ########

# name: (xmin, xmax, ymin, ymax)
domains = {
    # 'n_australia': (130, 153, -25, 5), # PNG and north Australia
    # 'e_australia': (135, 155, -45, -10), # North coast + Great Dividing Range + Tasmania
    # 'se_australia': (142, 155, -45, -28), # Great Dividing Range + Tasmania
    'small_europe': (-10, 15, 33, 60), # centred on France
    # 'large_europe': (-40, 40, 20, 80),
}

# secondary low resolution data sources


######### Figure parameters ########

ZEX = 0.5        # vertical exaggeration: degrees of height per 1000 m
THICK = 1.0      # slab thickness [deg]
NLAYER = 5       # ocean column layers on slab walls
AZIM = 20        # view azimuth in (0, 90) deg: 45 is symmetric, lower shows more of the south face
ELEV = 0.5       # vertical foreshortening of the ground plane (0-1)

cmap = LinearSegmentedColormap.from_list(
    'lowgreen_highbrown_topsnow',
    ['#5fae4e', '#9cc85a', '#e6d777', '#d9a45c', '#b5703a', '#f4f6f8'])
OCEAN_TOP = '#bfe3f2'
OCEAN_WALL = ['#9fd0ea', '#7fbbe0', '#5da4d6', '#4390c9', '#2f7ab8']
LAND_WALL = '#c8793a'
VMAX = 2500      # elevation [m] at which the colour scale reaches full brown (same for all panels)

###################################

def load(res, domain):
    xmin, xmax, ymin, ymax = domain
    if res == 'n48e':
        orogpath = f'/g/data/access/TIDS/UM/ancil/atmos/{res}/orography/globe30/v2/qrparm.orog'
        lsmpath = f'/g/data/access/TIDS/UM/ancil/atmos/{res}/land_sea_mask/igbp/v2/qrparm.landfrac'
    else:
        orogpath = f'{ancildir}/{res}/mom025_20250315/orography/gmted_ramp2/qrparm.orog_mean'
        lsmpath = f'{ancildir}/{res}/mom025_20250315/land_sea_mask/etop01/qrparm.landfrac'
    orog = xr.DataArray.from_iris(iris.load_cube(orogpath, constraint='surface_altitude'))
    lsm = xr.DataArray.from_iris(iris.load_cube(lsmpath))
    orog = orog.where(lsm.data > 0.5)
    orog = orog.assign_coords(longitude=(orog.longitude - xmin) % 360 + xmin)
    orog = orog.sortby(['latitude', 'longitude']).sel(
        longitude=slice(xmin, xmax), latitude=slice(ymin, ymax))
    return orog.squeeze()


def edges(c):
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def proj(x, y, z):
    a = np.radians(AZIM)
    return np.column_stack([x * np.cos(a) - y * np.sin(a),
                            (x * np.sin(a) + y * np.cos(a)) * ELEV + z])


def build(orog, domain):
    """Return polygons and facecolours for a back-to-front render."""
    xmin, _, ymin, _ = domain
    lon = orog.longitude.values
    lat = orog.latitude.values
    xe = edges(lon) - xmin
    ye = edges(lat) - ymin
    h = np.nan_to_num(orog.values, nan=0.0)
    land = np.isfinite(orog.values)
    z = np.where(land, np.maximum(h, 0) / 1000 * ZEX, 0.0)
    ny, nx = z.shape

    polys, colors = [], []

    def add(pts, col):
        polys.append(proj(*np.array(pts).T))
        colors.append(col)

    # far cells first (large x+y), so near cells overdraw
    order = sorted(((i, j) for j in range(ny) for i in range(nx)),
                   key=lambda t: -(t[0] + t[1]))
    for i, j in order:
        x0, x1, y0, y1 = xe[i], xe[i + 1], ye[j], ye[j + 1]
        zt = z[j, i]
        col = cmap(np.clip(h[j, i] / VMAX, 0, 1)) if land[j, i] else OCEAN_TOP
        # south-facing wall
        zs = -THICK if j == 0 else z[j - 1, i]
        if j == 0 or zt > zs:
            wall(add, (x0, y0), (x1, y0), zs, zt, land[j, i], j == 0, col)
        # west-facing wall
        zw = -THICK if i == 0 else z[j, i - 1]
        if i == 0 or zt > zw:
            wall(add, (x0, y0), (x0, y1), zw, zt, land[j, i], i == 0, col)
        add([(x0, y0, zt), (x1, y0, zt), (x1, y1, zt), (x0, y1, zt)], col)
    return polys, colors


def wall(add, p0, p1, zlow, zhigh, is_land, is_edge, col):
    (xa, ya), (xb, yb) = p0, p1
    if is_edge and not is_land:
        # ocean column: flat surface down to slab base, in stacked layers
        zs = np.linspace(zhigh, zlow, NLAYER + 1)
        for k in range(NLAYER):
            add([(xa, ya, zs[k]), (xb, yb, zs[k]), (xb, yb, zs[k + 1]), (xa, ya, zs[k + 1])],
                OCEAN_WALL[k % len(OCEAN_WALL)])
    else:
        add([(xa, ya, zlow), (xb, yb, zlow), (xb, yb, zhigh), (xa, ya, zhigh)],
            LAND_WALL if is_edge else col)


def main():
    os.makedirs(plotpath, exist_ok=True)
    for domain_name, domain in domains.items():
        print(f'domain: {domain_name}')
        plot_domain(domain_name, domain)


def plot_domain(domain_name, domain):
    plt.close('all')
    fig, axs = plt.subplots(len(reses), 1, figsize=(5, 3.6 * len(reses)))
    lim = None
    for ax, res in zip(axs, reses):
        orog = load(res, domain)
        dlat = float(np.abs(np.diff(orog.latitude.values)).mean())
        res_km = int(round(dlat * 111.32))
        print(f'{res}: {orog.shape} cells, ~{res_km} km')

        polys, colors = build(orog, domain)
        n = max(orog.shape)
        lw = float(np.clip(20 / n, 0.03, 0.6))
        edge = (0, 0, 0, 0.9 if n < 100 else 0.5)
        pc = PolyCollection(polys, facecolors=colors, edgecolors=edge, linewidths=lw)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(0.0, 1.0, res, transform=ax.transAxes, fontsize=20, fontweight='bold',
                va='top')
        ax.text(0.97, 0.12, f'~{res_km} km', transform=ax.transAxes, fontsize=11,
                fontweight='bold', ha='right', va='top')
        # ax.text(0.97, 0.04, f'{orog.shape[1]}x{orog.shape[0]} cells', transform=ax.transAxes,
        #         fontsize=9, ha='right', va='top')

        # same extent on all panels
        pts = np.vstack(polys)
        cur = np.array([pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()])
        lim = cur if lim is None else lim
    for ax in axs:
        ax.set_xlim(lim[0] - 0.5, lim[1] + 0.5)
        ax.set_ylim(lim[2] - 0.5, lim[3] + 0.5)

    fig.tight_layout()
    fig.text(0.0, -0.01, 'm.lipson@unsw.edu.au for 21st Century Weather', color='0.5',
             fontsize=6, ha='left', va='bottom')
    fname = f'{plotpath}/resolutions_3d_{domain_name}.png'
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'figure saved: {fname}')

if __name__ == '__main__':
    main()
