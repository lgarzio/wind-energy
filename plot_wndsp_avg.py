#!/usr/bin/env python

"""
Author: Lori Garzio on 4/12/2021
Last modified: 4/13/2021
Plot average WRF windspeeds at 10m and 160m at user-defined grouping intervals
"""

import datetime as dt
import pandas as pd
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'
    heights = [160, 10]

    # define the subsetting for the quivers on the map based on model and height
    quiver_subset = dict(_3km=dict(_10m=11, _80m=12, _160m=13))

    # define the vmin and vmax based on height
    hlims = dict(_10m=dict(vmin=0, vmax=10),
                 _160m=dict(vmin=0, vmax=10))

    axis_limits = [-79.79, -69.2, 34.5, 43]  # axis limits for the 3km model
    xticks = [-78, -76, -74, -72, -70]
    yticks = [36, 38, 40, 42]
    color_label = 'Average Wind Speed (m/s)'

    la_polygon = cf.extract_lease_areas()

    savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    # break up date range into the plotting interval specified
    if intvl == 'monthly':
        daterange = pd.date_range(sdate, edate, freq='M')
        start = []
        end = []
        for i, dr in enumerate(daterange):
            if i == 0:
                start.append(sdate)
            else:
                start.append((daterange[i - 1] + dt.timedelta(days=1)))
            end.append(dr)
        if dr + dt.timedelta(days=1) != edate:
            start.append((dr + dt.timedelta(days=1)))
            end.append(edate)

    # grab the WRF data for each interval
    for sd, ed in zip(start, end):
        # sd = dt.datetime(2019, 9, 1, 0, 0)  # for debugging
        # ed = dt.datetime(2019, 9, 1, 0, 0)  # for debugging
        ds = xr.open_dataset(wrf)
        ds = ds.sel(time=slice(sd, ed + dt.timedelta(hours=23)))
        for height in heights:
            if height == 10:
                u = ds['U10']
                v = ds['V10']
            else:
                u = ds.sel(height=height)['U']
                v = ds.sel(height=height)['V']

            ws = cf.wind_uv_to_spd(u, v)
            mws = ws.mean('time')
            print('{} {}m: min = {}, max = {}'.format(sd.strftime('%Y%m%d'), height, np.nanmin(mws), np.nanmax(mws)))
            u_mean = u.mean('time')
            v_mean = v.mean('time')
            sname = 'meanws_{}m_{}_{}'.format(height, intvl, sd.strftime('%Y%m%d'))
            sfile = os.path.join(savedir, sname)
            ttl = 'Average Wind Speed: {}m \n{} to {}'.format(height, sd.strftime('%Y-%m-%d'), ed.strftime('%Y-%m-%d'))
            qs = quiver_subset['_3km']['_{}m'.format(height)]

            # set up the map
            fig, ax, lat, lon = pf.set_map(ws)
            pf.add_map_features(ax, axis_limits, xticks, yticks)

            # pf.add_lease_area_polygon(ax, la_polygon, 'magenta')

            # plot data
            # pcolormesh: coarser resolution, shows the actual resolution of the model data
            pf.plot_pcolormesh(fig, ax, ttl, lon, lat, mws, 0, 40, 'BuPu', color_label)

            # subset the quivers and add as a layer
            ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_mean.values[::qs, ::qs], v_mean.values[::qs, ::qs],
                      scale=1000, width=.002, headlength=4, transform=ccrs.PlateCarree())

            plt.savefig(sfile, dpi=200)
            plt.close()


if __name__ == '__main__':
    save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    start_date = dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 9, 1, 0, 0)
    interval = 'monthly'
    main(save_dir, start_date, end_date, interval)
