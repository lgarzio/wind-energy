#!/usr/bin/env python

"""
Author: Lori Garzio on 6/3/2022
Last modified: 6/3/2022
Plot 2m Air Temp
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as cmo
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(fdir, savedir, plot_turbs):
    files = sorted(glob.glob(fdir + '*.nc'))
    plt_region = cf.plot_regions('1km')
    extent = plt_region['windturb']['extent']
    xticks = plt_region['windturb']['xticks']
    yticks = plt_region['windturb']['yticks']
    plt_vars = ['T2']
    color_label = 'Air Temperature at 2m (\N{DEGREE SIGN}C)'

    for fname in files:
        for pv in plt_vars:

            splitter = fname.split('/')[-1].split('_')

            save_name = '{}_{}_{}_{}.png'.format(pv, fname.split('/')[-3], splitter[2], splitter[-1].split('.nc')[0])

            sdir = os.path.join(savedir, pv)
            save_file = os.path.join(sdir, save_name)
            os.makedirs(sdir, exist_ok=True)

            ds = xr.open_dataset(fname)

            # convert from K to C
            z = np.squeeze(ds[pv]) - 273.15

            levels = list(np.arange(14, 30, .5))

            # mask the land values
            landmask = ds['LANDMASK']  # 1=land, 0=water
            lakemask = ds['LAKEMASK']  # 1=lake, 0=non-lake

            ldmask = np.logical_and(landmask == 1, landmask == 1)
            z.values[ldmask] = np.nan

            lkmask = np.logical_and(lakemask == 1, lakemask == 1)
            z.values[lkmask] = np.nan

            lon = z.XLONG.values
            lat = z.XLAT.values

            # set up the map
            lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
            pf.add_map_features(ax, extent, xticks=xticks, yticks=yticks, landcolor='tan', zoom_shore=True)

            la_polygon, pa_polygon = cf.extract_lease_area_outlines()
            kwargs = dict()
            kwargs['lw'] = 1.2
            pf.add_lease_area_polygon(ax, la_polygon, 'magenta', **kwargs)  # lease areas  '#969696'  '#737373'

            # set color map
            cmap = cmo.cm.thermal

            kwargs = dict()
            kwargs['ttl'] = '{} {}'.format(color_label, pd.to_datetime(ds.Time.values[0]).strftime('%Y-%m-%d %H:%M'))
            kwargs['cmap'] = cmap
            kwargs['clab'] = color_label
            kwargs['levels'] = levels
            kwargs['extend'] = 'both'
            pf.plot_contourf(fig, ax, lon, lat, z, **kwargs)

            if plot_turbs:
                df = pd.read_csv(plot_turbs)
                ax.scatter(df.lon, df.lat, s=.5, color='k', transform=ccrs.PlateCarree())

            plt.savefig(save_file, dpi=200)
            plt.close()


if __name__ == '__main__':
    file_dir = '/home/lgarzio/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20200622/'  # server
    save_dir = '/www/home/lgarzio/public_html/bpu/windturbs/20200622/'  # server
    plot_turbines = '/www/home/lgarzio/public_html/bpu/windturbs/turbine_locations_final.csv'  # server
    # file_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20200622/'
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/20200622/'
    # plot_turbines = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/turbine_locations_final.csv'
    main(file_dir, save_dir, plot_turbines)
