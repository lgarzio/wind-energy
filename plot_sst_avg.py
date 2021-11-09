#!/usr/bin/env python

"""
Author: Lori Garzio on 11/9/2021
Last modified: 11/9/2021
Plot average WRF SST at user-defined grouping intervals (seabreeze vs non-seabreeze days)
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as cmo
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plot_average_sst(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None

    plt_regions = cf.plot_regions(interval_name)
    plt_vars = dict(sst_mean=dict(color_label='Average SST (\N{DEGREE SIGN}C)',
                                  title='Average Sea Surface Temperature (\N{DEGREE SIGN}C)',
                                  cmap=cmo.cm.thermal),
                    sst_sd=dict(color_label='Variance ($^\circ$C)',
                                title='SST Variance',
                                cmap=cmo.cm.thermal)
                    )

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    # convert K to C
    sst = ds_sub.SST - 273.15
    sst_mean = sst.mean('time')
    sst_std = sst.std('time')

    landmask = ds_sub.LANDMASK.mean('time')  # 1=land, 0=water
    lakemask = ds_sub.LAKEMASK.mean('time')  # 1=lake, 0=non-lake

    plt_vars['sst_mean']['data'] = sst_mean
    plt_vars['sst_sd']['data'] = sst_std

    for pv, plt_info in plt_vars.items():
        for pr, region_info in plt_regions.items():
            region_savedir = os.path.join(save_dir, pr)
            os.makedirs(region_savedir, exist_ok=True)

            sname = '{}_{}_{}'.format(pr, pv, interval_name)
            if interval_name == 'seabreeze_days':
                nm = 'Sea breeze days'
            elif interval_name == 'noseabreeze_days':
                nm = 'Non-sea breeze days'
            elif interval_name == 'seabreeze_morning':
                nm = 'Sea breeze days (00Z - 13Z)'
            elif interval_name == 'seabreeze_afternoon':
                nm = 'Sea breeze days (14Z - 23Z)'
            elif interval_name == 'noseabreeze_morning':
                nm = 'Non-sea breeze days (00Z - 13Z)'
            elif interval_name == 'noseabreeze_afternoon':
                nm = 'Non-sea breeze days (14Z - 23Z)'
            elif interval_name == 'summer2020_all':
                nm = 'Overall'
            ttl = '{}\n{}\n{} to {}'.format(plt_info['title'], nm, sb_t0str, sb_t1str)
            sfile = os.path.join(region_savedir, sname)

            # set up the map
            lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
            data = plt_info['data']

            # map features
            kwargs = dict()
            kwargs['landcolor'] = 'tan'
            kwargs['xticks'] = region_info['xticks']
            kwargs['yticks'] = region_info['yticks']
            pf.add_map_features(ax, region_info['extent'], **kwargs)

            # subset grid
            if region_info['subset']:
                extent = np.add(region_info['extent'], [-.5, .5, -.5, .5]).tolist()
                data = cf.subset_grid(data, extent)
                landmask = cf.subset_grid(landmask, extent)
                lakemask = cf.subset_grid(lakemask, extent)

            # add lease areas
            if region_info['lease_area']:
                pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas
                pf.add_lease_area_polygon(ax, pa_polygon, '#969696')  # planning areas
                #leasing_areas.plot(ax=ax, lw=.8, color='magenta', transform=ccrs.LambertConformal())

            lon = data.XLONG.values
            lat = data.XLAT.values

            # mask values over land and lakes
            ldmask = np.logical_and(landmask == 1, landmask == 1)
            data.values[ldmask] = np.nan

            lkmask = np.logical_and(lakemask == 1, lakemask == 1)
            data.values[lkmask] = np.nan

            # initialize keyword arguments for plotting
            kwargs = dict()

            # plot data
            # pcolormesh: coarser resolution, shows the actual resolution of the model data
            # contourf: smooths the resolution of the model data, plots are less pixelated, can define discrete levels
            try:
                vmin = region_info[pv]['limits']['vmin']
                vmax = region_info[pv]['limits']['vmax']
                arange_interval = region_info[pv]['limits']['rint']
                levels = list(np.arange(vmin, vmax + arange_interval, arange_interval))
                kwargs['levels'] = levels
            except KeyError:
                print('no levels specified')

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
    # ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 5, 0)))  # for debugging

    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    # break up dates into the plotting interval specified
    if 'seabreeze' in intvl:
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # grab the WRF data for the seabreeze dates
        ds_sb = ds.sel(time=sb_datetimes)
        # ds_sb = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 15, 0)))  # for debugging

        # grab the WRF data for the non-seabreeze dates
        nosb_datetimes = [t for t in ds.time.values if t not in sb_datetimes]
        ds_nosb = ds.sel(time=nosb_datetimes)
        # ds_nosb = ds.sel(time=slice(dt.datetime(2020, 6, 2, 0, 0), dt.datetime(2020, 6, 2, 15, 0)))  # for debugging

        plot_average_sst(ds_sb, savedir, 'seabreeze_days', **kwargs)
        plot_average_sst(ds_nosb, savedir, 'noseabreeze_days', **kwargs)

    else:
        plot_average_sst(ds, savedir, intvl, **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'seabreeze_days'  # 'seabreeze_days' 'summer2020_all'
    main(save_directory, start_date, end_date, interval)
