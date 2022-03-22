#!/usr/bin/env python

"""
Author: Lori Garzio on 2/16/2022
Last modified: 2/16/2022
Plot the center grid points near the Atlantic City WEA
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(ncf, save_file):
    ds = xr.open_dataset(ncf)
    extent = [-74.6, -73.8, 38.85, 39.7]

    lon = ds.XLONG.values
    lat = ds.XLAT.values

    # set up the map
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
    pf.add_map_features(ax, extent, zoom_shore=True)

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()
    #pf.add_lease_area_polygon(ax, la_polygon[0], '#737373')  # lease areas
    pf.add_lease_area_polygon_single(ax, la_polygon[1], '#737373')  # lease area by Atlantic City
    pf.add_lease_area_polygon_single(ax, la_polygon[9], '#737373')  # cut out of lease area by Atlantic City

    # # plot all points
    # ax.scatter(lon, lat, s=.25, transform=ccrs.PlateCarree())
    #
    # plt.savefig(save_file, dpi=200)
    # plt.close()

    # plot every other grid point within specific section of WEA
    mask = np.ones(np.shape(lon), dtype=bool)
    lapoly = Polygon(la_polygon[1])
    for i, loni in enumerate(lon):
        for j, lonj in enumerate(lon[i]):
            if lapoly.contains(Point(lon[i, j], lat[i, j])):
                mask[i, j] = False
            if lon[i, j] < -74.38:
                mask[i, j] = True
            if np.logical_or(lat[i, j] > 39.365, lat[i, j] < 39):
                mask[i, j] = True
            if np.logical_and(lon[i, j] < -74.27, lat[i, j] > 39.2):
                mask[i, j] = True
            if np.logical_and(lon[i, j] < -74.34, lat[i, j] > 39.16):
                mask[i, j] = True
            if np.logical_and(lon[i, j] < -74.2, lat[i, j] > 39.22):
                mask[i, j] = True
            if np.logical_and(i % 2 == 0, j % 2 == 0):  # mask every other remaining grid point
                mask[i, j] = True
            if i % 2 != 0:
                mask[i, j] = True

    lon[mask] = np.nan
    lat[mask] = np.nan

    # export turbine locations as csv
    turbine_locs = dict(lon=[], lat=[])
    for i, loni in enumerate(lon):
        for j, lonj in enumerate(lon[i]):
            if ~np.isnan(lon[i, j]):
                turbine_locs['lon'].append(lon[i, j])
                turbine_locs['lat'].append(lat[i, j])

    pd.DataFrame(turbine_locs).to_csv('/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/turbine_locations.csv',
                                      index=False)

    ax.scatter(lon, lat, s=.25, transform=ccrs.PlateCarree())
    plt.savefig('/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/wrf_1km_grid_points-turbines.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20220116/wrfproc_1km_20220116_00Z_H000.nc'
    savefile = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/wrf_1km_grid_points.png'
    main(ncfile, savefile)
