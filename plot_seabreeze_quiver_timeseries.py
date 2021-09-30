#!/usr/bin/env python

"""
Author: Lori Garzio on 9/21/2021
Last modified: 9/30/2021
Quiver plots of hourly-averaged winds in Southern NJ over land and over water
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plot_feather(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250, 200, 160, 10]

    # extent = [-74.9, -73.9, 38.87, 39.7]  # test 1
    extent = [-74.9, -73.9, 38.87, 39.8]  # test2
    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    for h in heights:
        if h == 10:
            u = ds_sub['U10']
            v = ds_sub['V10']
        else:
            u = ds_sub.sel(height=h)['U']
            v = ds_sub.sel(height=h)['V']

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
            mask = cf.subset_grid_preserve_time(ds_sub['LANDMASK'], extent)
            mask_sub = np.logical_not(mask == value)
            u_sub.values[mask_sub.values] = np.nan
            v_sub.values[mask_sub.values] = np.nan

            # calculate hourly averages
            for hour in hours:
                u_hourly = u_sub[u_sub.time.dt.hour == hour].mean()
                v_hourly = v_sub[v_sub.time.dt.hour == hour].mean()
                data[lw]['u_hourly_mean'] = np.append(data[lw]['u_hourly_mean'], u_hourly)
                data[lw]['v_hourly_mean'] = np.append(data[lw]['v_hourly_mean'], v_hourly)

        # plot
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(12, 6))

        ax1.quiver(hours, 0, data['land']['u_hourly_mean'], data['land']['v_hourly_mean'])
        ax2.quiver(hours, 0, data['water']['u_hourly_mean'], data['water']['v_hourly_mean'])

        ax1.set_title(f'Hourly averages over land: {sb_t0str} to {sb_t1str} ({interval_name})')
        ax2.set_title(f'Hourly averages over water: {sb_t0str} to {sb_t1str} ({interval_name})')

        # fig.suptitle()
        ax2.set_xlabel('Hour')

        sname = f'timeseries_quiver_{interval_name}_{h}m.png'
        plt.savefig(os.path.join(save_dir, sname), dpi=200)
        plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    savedir = os.path.join(sDir, '{}_{}_{}-{}'.format('feather', intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    # ds = ds.sel(time=slice(sdate, edate))
    ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 2, 5, 0)))  # for debugging
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    if intvl == 'all':
        plot_feather(ds, savedir, intvl, **kwargs)

    elif intvl == 'seabreeze_days':
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in
                        sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # grab the WRF data for the seabreeze dates
        ds_sb = ds.sel(time=sb_datetimes)
        # ds_sb = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 15, 0)))  # for debugging

        # grab the WRF data for the non-seabreeze dates
        nosb_datetimes = [t for t in ds.time.values if t not in sb_datetimes]
        ds_nosb = ds.sel(time=nosb_datetimes)
        # ds_nosb = ds.sel(time=slice(dt.datetime(2020, 6, 2, 0, 0), dt.datetime(2020, 6, 2, 15, 0)))  # for debugging

        plot_feather(ds_sb, savedir, 'seabreeze_days', **kwargs)
        plot_feather(ds_nosb, savedir, 'noseabreeze_days', **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'seabreeze_days'  # all seabreeze_days
    main(save_directory, start_date, end_date, interval)
