#!/usr/bin/env python

"""
Author: Lori Garzio on 2/17/2022
Last modified: 2/22/2022
Plot the Atlantic City WEA and experimental wind turbine locations at every other grid point
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(f, save_file):
    df = pd.read_csv(f)
    extent = [-74.6, -73.8, 38.85, 39.7]

    # set up the map
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
    pf.add_map_features(ax, extent, landcolor='tan', zoom_shore=True)

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()
    pf.add_lease_area_polygon_single(ax, la_polygon[1], '#737373')  # lease area by Atlantic City
    pf.add_lease_area_polygon_single(ax, la_polygon[9], '#737373')  # cut out of lease area by Atlantic City

    # plot all points
    ax.scatter(df.lon, df.lat, s=2.5, transform=ccrs.PlateCarree())

    plt.savefig(save_file, dpi=200)
    plt.close()


if __name__ == '__main__':
    file = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/turbine_locations_final.csv'
    savefile = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/wrf_1km_turbine_locations_final.png'
    main(file, savefile)
