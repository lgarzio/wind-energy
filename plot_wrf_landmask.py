#!/usr/bin/env python

"""
Author: Lori Garzio on 9/29/2021
Last modified: 9/29/2021
Plot the WRF landmask
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(savedir):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=dt.datetime(2020, 6, 1, 0, 0))
    #extent = [-74.9, -73.9, 38.87, 39.7]  # test 1
    extent = [-74.9, -73.9, 38.87, 39.8]  # test2
    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
    pf.add_map_features(ax, [-75, -73.6, 38.7, 39.9])
    cmaps = ['Greens', 'Blues']
    pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas

    cnt = 0
    title = 'Distance from corner to coastline (cyan point)'

    # mask points over land/water
    landwater = dict(land=1, water=0)
    for lw, value in landwater.items():
        mask = cf.subset_grid(ds['LANDMASK'], extent)
        mask_sub = np.logical_not(mask == value)
        mask.values[mask_sub.values] = np.nan

        lon = mask.XLONG.values
        lat = mask.XLAT.values

        ax.contourf(lon, lat, mask, cmap=cmaps[cnt], transform=ccrs.PlateCarree())

        # plot the center point from which distance to edges of region are calculated
        lons = [np.nanmin(lon), np.nanmax(lon)]
        lats = [np.nanmin(lat), np.nanmax(lat)]
        # ax.plot([lons[0], lons[1]], [lats[0], lats[1]], 'r-', transform=ccrs.PlateCarree())
        # ax.plot([lons[0], lons[1]], [lats[1], lats[0]], 'r-', transform=ccrs.PlateCarree())
        midpoint = [-74.425, 39.355]
        ax.plot(midpoint[0], midpoint[1], 'cyan', marker='o', transform=ccrs.PlateCarree(), zorder=10)

        if lw == 'land':
            edge = [lons[0], lats[1]]
        else:
            edge = [lons[1], lats[0]]

        # calculate distance from region corner to mid-point
        geod = Geodesic.WGS84
        g = geod.Inverse(midpoint[1], midpoint[0],
                         edge[1], edge[0])
        distance_km = g['s12'] * .001

        #ax.plot([midpoint[0], edge[0]], [midpoint[1], edge[1]], 'r-', transform=ccrs.PlateCarree())

        title = f'{title}\n{lw}: {np.round(distance_km, 2)} km'
        cnt += 1

    plt.title(title, fontsize=16)
    plt.savefig(os.path.join(savedir, f'seabreeze_feather_regions.png'), dpi=200)
    plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    # save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    main(save_directory)
