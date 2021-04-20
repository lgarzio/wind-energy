#!/usr/bin/env python

"""
Author: Lori Garzio on 4/12/2021
Last modified: 4/20/2021
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
    hlims = dict(_10m=dict(vmin=4, vmax=10),
                 _160m=dict(vmin=8, vmax=14))

    axis_limits = [-79.79, -69.2, 34.5, 43]  # axis limits for the 3km model
    xticks = [-78, -76, -74, -72, -70]
    yticks = [36, 38, 40, 42]
    color_label = 'Average Wind Speed (m/s)'

    la_polygon = cf.extract_lease_areas()

    savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    # break up date range into the plotting interval specified
    start, end = cf.daterange_interval(intvl, sdate, edate)

    ds = xr.open_dataset(wrf)
    summary = []
    # grab the WRF data for each interval
    for sd, ed in zip(start, end):
        # sd = dt.datetime(2019, 9, 1, 0, 0)  # for debugging
        # ed = dt.datetime(2019, 9, 2, 0, 0)  # for debugging
        dst = ds.sel(time=slice(sd, ed + dt.timedelta(hours=23)))
        for height in heights:
            if height == 10:
                u = dst['U10']
                v = dst['V10']
            else:
                u = dst.sel(height=height)['U']
                v = dst.sel(height=height)['V']

            ws = cf.wind_uv_to_spd(u, v)
            mws = ws.mean('time')

            u_mean = u.mean('time')
            v_mean = v.mean('time')

            # standardize the vectors so they only represent direction
            u_mean_standardize = u_mean / cf.wind_uv_to_spd(u_mean, v_mean)
            v_mean_standardize = v_mean / cf.wind_uv_to_spd(u_mean, v_mean)

            sname = 'meanws_{}m_{}_{}'.format(height, intvl, sd.strftime('%Y%m%d'))
            sfile = os.path.join(savedir, sname)
            ttl = 'Average Wind Speed {}m: {}'.format(height, sd.strftime('%b %Y'))
            qs = quiver_subset['_3km']['_{}m'.format(height)]

            # set up the map
            fig, ax, lat, lon = pf.set_map(ws)
            pf.add_map_features(ax, axis_limits, xticks, yticks)

            # pf.add_lease_area_polygon(ax, la_polygon, 'magenta')

            # plot data
            # pcolormesh: coarser resolution, shows the actual resolution of the model data
            vmin = hlims['_{}m'.format(height)]['vmin']
            vmax = hlims['_{}m'.format(height)]['vmax']
            pf.plot_pcolormesh(fig, ax, ttl, lon, lat, mws, vmin, vmax, 'BuPu', color_label)

            # subset the quivers and add as a layer
            # ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_mean_standardize.values[::qs, ::qs], v_mean_standardize.values[::qs, ::qs],
            #           scale=100, width=.002, headlength=4, transform=ccrs.PlateCarree())

            ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_mean_standardize.values[::qs, ::qs],
                      v_mean_standardize.values[::qs, ::qs], transform=ccrs.PlateCarree())

            plt.savefig(sfile, dpi=200)
            plt.close()

            summary.append([sd.strftime('%Y-%d-%mT%H'), ed.strftime('%Y-%d-%mT%H'), height, np.nanmin(mws), np.nanmax(mws)])

    df = pd.DataFrame(summary, columns=['start', 'end', 'height_m', 'min_avgws', 'max_avgws'])
    df.to_csv(os.path.join(savedir, 'meanws_summary_{}_{}.csv'.format(intvl, sd.strftime('%Y%m%d'))), index=False)


if __name__ == '__main__':
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_dir = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 9, 1, 0, 0)
    interval = 'monthly'
    main(save_dir, start_date, end_date, interval)
