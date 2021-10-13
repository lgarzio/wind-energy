#!/usr/bin/env python

"""
Author: Lori Garzio on 10/12/2021
Last modified: 10/12/2021
Quiver plots of hourly-averaged winds at specific locations
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

    hours = np.arange(1, 24)

    # set up map
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig1, axm = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
    pf.add_map_features(axm, [-75, -73.6, 38.7, 39.9])
    la_polygon, pa_polygon = cf.extract_lease_area_outlines()
    pf.add_lease_area_polygon(axm, la_polygon, '#737373')  # lease areas

    # for each height, select the 3x3 box around the specified point
    for h in heights:
        points = dict(pt1=dict(loc=[-74.745, 39.64], u_hourly_mean=np.array([]), v_hourly_mean=np.array([])),
                      pt2=dict(loc=[-74.52, 39.44], u_hourly_mean=np.array([]), v_hourly_mean=np.array([])),
                      pt3=dict(loc=[-74.34, 39.275], u_hourly_mean=np.array([]), v_hourly_mean=np.array([])),
                      pt4=dict(loc=[-74.15, 39.1], u_hourly_mean=np.array([]), v_hourly_mean=np.array([])))
        for key, info in points.items():
            # calculate the sum of the absolute value distance between the model location and point
            a = abs(ds_sub.XLAT - info['loc'][1]) + abs(ds_sub.XLONG - info['loc'][0])

            # find the indices of the minimum value in the array calculated above
            i, j = np.unravel_index(a.argmin(), a.shape)

            # subset data around specified point
            if h == 10:
                u = ds_sub['U10'][:, (i - 1):(i + 1), (j - 1):(j + 1)]
                v = ds_sub['V10'][:, (i - 1):(i + 1), (j - 1):(j + 1)]
            else:
                u = ds_sub.sel(height=h)['U'][:, (i - 1):(i + 1), (j - 1):(j + 1)]
                v = ds_sub.sel(height=h)['V'][:, (i - 1):(i + 1), (j - 1):(j + 1)]

            # add points to map
            mask = np.empty(np.shape(u.XLONG.values))
            mask[:] = 1
            axm.contourf(u.XLONG.values, u.XLAT.values, mask, cmap='Greys', transform=ccrs.PlateCarree())
            axm.text(info['loc'][0], info['loc'][1], key, transform=ccrs.PlateCarree())

            # calculate hourly averages
            for hour in hours:
                u_hourly = u[u.time.dt.hour == hour].mean()
                v_hourly = v[v.time.dt.hour == hour].mean()
                points[key]['u_hourly_mean'] = np.append(points[key]['u_hourly_mean'], u_hourly)
                points[key]['v_hourly_mean'] = np.append(points[key]['v_hourly_mean'], v_hourly)

        # plot quiver timeseries
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey=True, sharex=True, figsize=(12, 8))

        ax1.quiver(hours, 0, points['pt1']['u_hourly_mean'], points['pt1']['v_hourly_mean'])
        ax2.quiver(hours, 0, points['pt2']['u_hourly_mean'], points['pt2']['v_hourly_mean'])
        ax3.quiver(hours, 0, points['pt3']['u_hourly_mean'], points['pt3']['v_hourly_mean'])
        ax4.quiver(hours, 0, points['pt4']['u_hourly_mean'], points['pt4']['v_hourly_mean'])

        ax1.set_title(f'{sb_t0str} to {sb_t1str}: {h}m ({interval_name})\nHourly averages point 1')
        ax2.set_title('point 2')
        ax3.set_title('point 3')
        ax4.set_title('point 4')

        # fig.suptitle()
        ax4.set_xlabel('Hour')

        sname = f'timeseries_quiver_points_{interval_name}_{h}m.png'
        fig.savefig(os.path.join(save_dir, sname), dpi=200)
        plt.close()

    # save the map
    sname = f'timeseries_quiver_map.png'
    fig1.savefig(os.path.join(save_dir, sname), dpi=200)
    plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    savedir = os.path.join(sDir, '{}_{}_{}-{}'.format('feather_points', intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))
    # ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 2, 5, 0)))  # for debugging
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
    interval = 'all'  # all seabreeze_days
    main(save_directory, start_date, end_date, interval)
