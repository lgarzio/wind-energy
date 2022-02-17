#!/usr/bin/env python

"""
Author: Lori Garzio on 2/16/2022
Last modified: 2/16/2022
Plot the center grid points near the Atlantic City WEA
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
    pf.add_map_features(ax, extent)

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()
    pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas
    pf.add_lease_area_polygon(ax, pa_polygon, '#969696')  # planning areas

    ax.scatter(lon, lat, s=.25, transform=ccrs.PlateCarree())

    plt.savefig(save_file, dpi=200)
    plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20220116/wrfproc_1km_20220116_00Z_H000.nc'
    savefile = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/wrf_1km_grid_points.png'
    main(ncfile, savefile)
