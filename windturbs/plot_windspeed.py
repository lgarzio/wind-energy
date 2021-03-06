#!/usr/bin/env python

"""
Author: Lori Garzio on 2/16/2022
Last modified: 2/22/2022
Plot wind speed at hub height (160m)
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(fdir, savedir, plot_turbs):
    files = sorted(glob.glob(fdir + '*.nc'))
    plt_region = cf.plot_regions('1km')
    extent = plt_region['windturb']['extent']
    xticks = plt_region['windturb']['xticks']
    yticks = plt_region['windturb']['yticks']
    color_label = 'Wind Speed (m/s)'
    heights = [160, 10]

    for fname in files:

        splitter = fname.split('/')[-1].split('_')

        for ht in heights:

            save_name = 'windspeed_{}m_{}_{}_{}.png'.format(ht, fname.split('/')[-3], splitter[2], splitter[-1].split('.nc')[0])

            sdir = os.path.join(savedir, f'windspeed_{ht}m')
            save_file = os.path.join(sdir, save_name)
            os.makedirs(sdir, exist_ok=True)

            ds = xr.open_dataset(fname)

            if ht == 10:
                u = np.squeeze(ds['U10'])
                v = np.squeeze(ds['V10'])
            else:
                u = np.squeeze(ds.sel(height=ht)['U'])
                v = np.squeeze(ds.sel(height=ht)['V'])

            # standardize the vectors so they only represent direction
            u_standardize = u / cf.wind_uv_to_spd(u, v)
            v_standardize = v / cf.wind_uv_to_spd(u, v)

            # calculate wind speed from u and v
            speed = cf.wind_uv_to_spd(u, v)
            lon = speed.XLONG.values
            lat = speed.XLAT.values

            # set up the map
            lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
            pf.add_map_features(ax, extent, xticks=xticks, yticks=yticks, zoom_shore=True)

            la_polygon, pa_polygon = cf.extract_lease_area_outlines()
            kwargs = dict()
            kwargs['lw'] = 1.2
            pf.add_lease_area_polygon(ax, la_polygon, 'magenta', **kwargs)  # lease areas  '#969696'  '#737373'

            # set color map
            cmap = plt.get_cmap('BuPu')
            levels = list(np.arange(2, 16.5, .5))

            kwargs = dict()
            kwargs['ttl'] = '{} {}'.format(color_label, pd.to_datetime(ds.Time.values[0]).strftime('%Y-%m-%d %H:%M'))
            kwargs['cmap'] = cmap
            kwargs['clab'] = color_label
            kwargs['levels'] = levels
            kwargs['extend'] = 'both'
            pf.plot_contourf(fig, ax, lon, lat, speed, **kwargs)

            qs = 5
            ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_standardize.values[::qs, ::qs], v_standardize.values[::qs, ::qs],
                      scale=30, width=.002, headlength=4, transform=ccrs.PlateCarree())

            if plot_turbs:
                df = pd.read_csv(plot_turbs)
                ax.scatter(df.lon, df.lat, s=.5, color='k', transform=ccrs.PlateCarree())

            plt.savefig(save_file, dpi=200)
            plt.close()


if __name__ == '__main__':
    file_dir = '/home/lgarzio/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20210902/'  # server
    save_dir = '/www/home/lgarzio/public_html/bpu/windturbs/20210902/'  # server
    plot_turbines = '/www/home/lgarzio/public_html/bpu/windturbs/turbine_locations_final.csv'  # server
    # file_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20210901/'
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/20210901/'
    # plot_turbines = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/turbine_locations_final.csv'
    main(file_dir, save_dir, plot_turbines)
