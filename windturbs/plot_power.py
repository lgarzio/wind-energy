#!/usr/bin/env python

"""
Author: Lori Garzio on 2/16/2022
Last modified: 12/1/2022
Plot power from simulated wind turbines in RU-WRF
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(fdir, savedir):
    files = sorted(glob.glob(os.path.join(fdir, '*.nc')))
    if files[0].endswith('_M00.nc'):
        files = sorted(glob.glob(os.path.join(fdir, '*_M00.nc')))
    plt_region = cf.plot_regions('1km')
    extent = plt_region['windturb']['extent']
    xticks = plt_region['windturb']['xticks']
    yticks = plt_region['windturb']['yticks']
    color_label = 'Wind Power (kW)'

    for fname in files:
        run_type = fname.split('/')[-4]
        if 'ctrl' in run_type:
            plot_turbs = False
        else:
            plot_turbs = '/www/web/rucool/windenergy/ru-wrf/windturbs/turbine_locations_final.csv'  # server
            #plot_turbs = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/turbine_locations_final.csv'

        ds = xr.open_dataset(fname)
        tm = pd.to_datetime(ds.Time.values[0])

        save_name = 'power_{}_{}_H{:03d}.png'.format(run_type, tm.strftime('%Y%m%d'), tm.hour)

        sdir = os.path.join(savedir, tm.strftime('%Y%m%d'), 'power')
        save_file = os.path.join(sdir, save_name)
        os.makedirs(sdir, exist_ok=True)

        power = np.squeeze(ds.POWER) / 1000  # power in kW

        mask = np.logical_and(power == 0, power == 0)
        power.values[mask] = np.nan

        lon = power.XLONG.values
        lat = power.XLAT.values

        # set up the map
        lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
        pf.add_map_features(ax, extent, xticks=xticks, yticks=yticks, landcolor='tan', zoom_shore=True)

        la_polygon, pa_polygon = cf.extract_lease_area_outlines()
        pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas

        # set color map
        cmap = plt.get_cmap('OrRd')
        levels = list(np.arange(0, 15001, 1000))

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        kwargs = dict()
        kwargs['ttl'] = '{} {}'.format(color_label, pd.to_datetime(ds.Time.values[0]).strftime('%Y-%m-%d %H:%M'))
        kwargs['cmap'] = cmap
        kwargs['clab'] = color_label
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'neither'
        pf.plot_pcolormesh(fig, ax, lon, lat, power, **kwargs)

        # add power values of 15000 as another layer
        power_copy = power.copy()
        custom_color = ["#67000d"]  # dark red
        custom_colormap = ListedColormap(custom_color)
        mask = np.logical_and(power_copy.values < 15001, power_copy.values < 15001)
        power_copy.values[mask] = np.nan
        ax.pcolormesh(lon, lat, power_copy, cmap=custom_colormap, transform=ccrs.PlateCarree())

        if plot_turbs:
            df = pd.read_csv(plot_turbs)
            ax.scatter(df.lon, df.lat, s=.5, color='k', transform=ccrs.PlateCarree())

        plt.savefig(save_file, dpi=200)
        plt.close()


if __name__ == '__main__':
    file_dir = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout_windturbs/constdt/1km_wf2km/processed/20220810'  # server
    save_dir = '/www/web/rucool/windenergy/ru-wrf/windturbs'  # server
    #file_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20220810'
    #save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/20220810/'
    main(file_dir, save_dir)
