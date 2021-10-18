#!/usr/bin/env python

"""
Author: Lori Garzio on 10/18/2021
Last modified: 10/18/2021
Plot hourly scalar product at multiple heights
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


def plot_scalar(ds_sub, ds_sb_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250, 200, 160, 10]

    plt_regions = cf.plot_regions(interval_name)
    plt_vars = dict(scalar=dict(color_label='Hourly-Averaged Scalar Product',
                                title='Hourly-Averaged Scalar Product',
                                cmap=plt.get_cmap('RdBu')))

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    for height in heights:
        if height == 10:
            u = ds_sub['U10']
            v = ds_sub['V10']
            u_sb = ds_sb_sub['U10']
            v_sb = ds_sb_sub['V10']
        else:
            u = ds_sub.sel(height=height)['U']
            v = ds_sub.sel(height=height)['V']
            u_sb = ds_sb_sub.sel(height=height)['U']
            v_sb = ds_sb_sub.sel(height=height)['V']

        hours = np.arange(1, 25)

        # calculate hourly averages
        for hour in hours:
            # calculate hourly averages - seabreeze days only
            u_hour_sb = u_sb[u_sb.time.dt.hour == hour]
            v_hour_sb = v_sb[v_sb.time.dt.hour == hour]
            ws_hour_sb = cf.wind_uv_to_spd(u_hour_sb, v_hour_sb)
            ws_hourly_mean_sb = ws_hour_sb.mean('time')  # A

            # calculate hourly averages - all days
            u_hour = u[u.time.dt.hour == hour]
            v_hour = v[v.time.dt.hour == hour]
            ws_hour = cf.wind_uv_to_spd(u_hour, v_hour)
            ws_hourly_mean = ws_hour.mean('time')  # B

            # calculate hourly-averaged vectors for dot calculation - seabreeze days only
            u_hourly_mean_sb = u_hour_sb.mean('time')
            v_hourly_mean_sb = v_hour_sb.mean('time')

            # calculate hourly-averaged vectors for dot calculation - all days
            u_hourly_mean = u_hour.mean('time')
            v_hourly_mean = v_hour.mean('time')

            dot = (u_hourly_mean_sb * u_hourly_mean) + (v_hourly_mean_sb * v_hourly_mean)

            cos = dot / (ws_hourly_mean_sb * ws_hourly_mean)  # scalar product

            plt_vars['scalar']['data'] = cos

            for pv, plt_info in plt_vars.items():
                for pr, region_info in plt_regions.items():
                    region_savedir = os.path.join(save_dir, pr)
                    os.makedirs(region_savedir, exist_ok=True)

                    sname = '{}_{}m_{}_H{}'.format(pr, height, interval_name, str(hour).zfill(3))
                    ttl = '{} {}m: H{}\n{} to {}'.format(plt_info['title'], height, str(hour).zfill(3),
                                                         sb_t0str, sb_t1str)

                    sfile = os.path.join(region_savedir, sname)

                    # set up the map
                    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
                    pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'])

                    data = plt_info['data']

                    # subset grid
                    if region_info['subset']:
                        extent = np.add(region_info['extent'], [-.5, .5, -.5, .5]).tolist()
                        data = cf.subset_grid(data, extent)

                    # add lease areas
                    if region_info['lease_area']:
                        pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas
                        pf.add_lease_area_polygon(ax, pa_polygon, '#969696')  # planning areas
                        #leasing_areas.plot(ax=ax, lw=.8, color='magenta', transform=ccrs.LambertConformal())

                    lon = data.XLONG.values
                    lat = data.XLAT.values

                    # initialize keyword arguments for plotting
                    kwargs = dict()

                    # plot data
                    # pcolormesh: coarser resolution, shows the actual resolution of the model data
                    # contourf: smooths the resolution of the model data, plots are less pixelated, can define discrete levels
                    #kwargs['levels'] = list(np.arange(-1, 1.25, .25))
                    kwargs['levels'] = np.linspace(-1, 1, 21)
                    kwargs['extend'] = 'neither'

                    kwargs['ttl'] = ttl
                    kwargs['clab'] = plt_info['color_label']
                    pf.plot_contourf(fig, ax, lon, lat, data, plt_info['cmap'], **kwargs)

                    plt.savefig(sfile, dpi=200)
                    plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))

    # get just the seabreeze days dataset
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')
    df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
    df = df[df['Seabreeze'] == 'y']
    sb_dates = np.array(pd.to_datetime(df['Date']))
    sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
    sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    # grab the WRF data for the seabreeze dates
    ds_sb = ds.sel(time=sb_datetimes)
    # ds_sb = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 15, 0)))  # for debugging
    # ds = ds.sel(time=slice(dt.datetime(2020, 6, 2, 0, 0), dt.datetime(2020, 6, 2, 15, 0)))  # for debugging

    plot_scalar(ds, ds_sb, savedir, intvl, **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'scalar_hourly'
    main(save_directory, start_date, end_date, interval)
