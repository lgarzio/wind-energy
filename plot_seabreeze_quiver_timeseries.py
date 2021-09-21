#!/usr/bin/env python

"""
Author: Lori Garzio on 9/21/2021
Last modified: 9/21/2021
Quiver plots of hourly-averaged winds in Southern NJ over land and over water
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions.common as cf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(sDir, sdate, edate, hts):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    savedir = os.path.join(sDir, '{}_{}-{}'.format('feather', sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))
    #ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 2, 5, 0)))  # for debugging
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')
    extent = [-74.9, -73.8, 38.8, 39.8]
    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    for h in hts:
        if h == 10:
            u = ds['U10']
            v = ds['V10']
        else:
            u = ds.sel(height=h)['U']
            v = ds.sel(height=h)['V']

        hours = np.arange(1, 24)

        # mask points over land/water
        data = dict()
        landwater = dict(land=1, water=0)
        for lw, value in landwater.items():
            data[lw] = dict()
            data[lw]['hours'] = hours
            data[lw]['u_hourly_mean'] = np.array([])
            data[lw]['v_hourly_mean'] = np.array([])
            u_sub = cf.subset_grid_preserve_time(u, extent)
            v_sub = cf.subset_grid_preserve_time(v, extent)
            mask = cf.subset_grid_preserve_time(ds['LANDMASK'], extent)
            mask_sub = np.logical_not(mask == value)
            u_sub.values[mask_sub.values] = np.nan
            v_sub.values[mask_sub.values] = np.nan

            ws = cf.wind_uv_to_spd(u_sub, v_sub)
            # mws = ws.mean('time')
            #
            # lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
            # pf.add_map_features(ax, [-75, -73.6, 38.7, 39.9])
            #
            # lon = u_sub.XLONG.values
            # lat = u_sub.XLAT.values
            # kwargs = dict()
            #
            # pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas
            # pf.plot_contourf(fig, ax, lon, lat, mws, 'BuPu', **kwargs)
            #
            # plt.savefig(os.path.join(savedir, f'seabreeze_feather_{lw}.png'), dpi=200)
            # plt.close()

            # calculate hourly averages
            for hour in hours:
                u_hourly = u_sub[u_sub.time.dt.hour == hour].mean()
                v_hourly = v_sub[v_sub.time.dt.hour == hour].mean()
                data[lw]['u_hourly_mean'] = np.append(data[lw]['u_hourly_mean'], u_hourly)
                data[lw]['v_hourly_mean'] = np.append(data[lw]['v_hourly_mean'], v_hourly)

        # plot
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(12, 6))

        ax1.quiver(hours, 0, data['land']['u_hourly_mean'], data['land']['v_hourly_mean'],
                   units='y', scale_units='y', scale=1, headlength=1, headaxislength=1, width=0.004)
        ax2.quiver(hours, 0, data['water']['u_hourly_mean'], data['water']['v_hourly_mean'])

        ax1.set_title('Land')
        ax2.set_title('Water')

        #fig.suptitle()
        ax2.set_xlabel('Hour')

        sname = f'timeseries_quiver_{h}m.png'
        plt.savefig(os.path.join(savedir, sname), dpi=200)
        plt.close()


if __name__ == '__main__':
    #save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    heights = [250, 200, 160, 10]
    main(save_directory, start_date, end_date, heights)
