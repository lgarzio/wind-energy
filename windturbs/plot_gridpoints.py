#!/usr/bin/env python

"""
Author: Lori Garzio on 3/22/2022
Last modified: 3/22/2022
Plot the center of the WRF grid points
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(ncf, save_file):
    oyster_creek = [39.811474, -74.20145]
    tuckerton_sodar = [39.5149, -74.307648]
    ds = xr.open_dataset(ncf)
    regions = cf.plot_regions()
    region = regions['southern_nj']
    extent = region['extent']

    lon = ds.XLONG.values
    lat = ds.XLAT.values

    # set up the map
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
    kwargs = dict()
    kwargs['xticks'] = region['xticks']
    kwargs['yticks'] = region['yticks']
    kwargs['zoom_shore'] = True
    pf.add_map_features(ax, extent, **kwargs)

    # plot the land mask
    landmask = np.squeeze(ds['LANDMASK'])  # 1=land, 0=water

    # ldmask = np.logical_and(landmask == 0, landmask == 0)
    # landmask.values[ldmask] = np.nan

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()
    kwargs = dict()
    kwargs['lw'] = 1.2
    pf.add_lease_area_polygon(ax, la_polygon, 'magenta', **kwargs)  # lease areas  '#969696'  '#737373'

    cmap = plt.get_cmap('Greys')
    cs = ax.pcolormesh(lon, lat, landmask, cmap=cmap, transform=ccrs.PlateCarree())

    # plot all grid points
    ax.scatter(lon, lat, s=.25, transform=ccrs.PlateCarree(), zorder=10)

    # add Oyster Creek and the Tuckerton SODAR stations
    ax.scatter(oyster_creek[1], oyster_creek[0], s=30, c='magenta', edgecolor='k', linewidth=.5,
               transform=ccrs.PlateCarree(), zorder=11)
    ax.scatter(tuckerton_sodar[1], tuckerton_sodar[0], s=30, c='magenta', edgecolor='k', linewidth=.5,
               transform=ccrs.PlateCarree(), zorder=11)

    plt.savefig(save_file, dpi=200)
    plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/bpu/wrf/wrf_files/processed/3km/20190901/wrfproc_3km_20190901_00Z_H000.nc'
    savefile = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/wrf_3km_grid_points.png'
    main(ncfile, savefile)
