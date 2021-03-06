#!/usr/bin/env python

"""
Author: Lori Garzio on 4/12/2021
Last modified: 4/8/2022
Plot average WRF windspeeds at 10m, 160m, 200m and 250m at user-defined grouping intervals (monthly and seabreeze vs
non-seabreeze days)
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


def plot_averages(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250, 200, 160, 10]
    mingray = dict(_10m=5, _160m=5.5, _200m=5.5, _250m=5.5)  # minimum average value for making the state/coastlines and quivers gray

    plt_regions = cf.plot_regions(interval_name)
    plt_vars = dict(meanpower=dict(color_label='Average Estimated 15MW Wind Power (kW)',
                                   title='Average Wind Power (15MW)',
                                   cmap='OrRd'),
                    sdpower=dict(color_label='Variance (kW)',
                                 title='Power Variance',
                                 cmap='OrRd'),
                    meanws=dict(color_label='Average Wind Speed (m/s)',
                                title='Average Wind Speed',
                                cmap=plt.get_cmap('BuPu')),
                    sdwind=dict(color_label='Variance (m/s)',
                                title='Wind Speed Variance',
                                cmap='BuPu'),
                    sdwind_norm=dict(color_label='Normalized Variance',
                                     title='Normalized Wind Speed Variance',
                                     cmap='BuPu'))

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    # for calculating power
    power_curve = pd.read_csv('/home/lgarzio/rucool/bpu/wrf/wrf_lw15mw_power.csv')  # on server
    #power_curve = pd.read_csv('/Users/garzio/Documents/rucool/bpu/wrf/wrf_lw15mw_power.csv')

    for height in heights:
        if height == 10:
            u = ds_sub['U10']
            v = ds_sub['V10']
        else:
            u = ds_sub.sel(height=height)['U']
            v = ds_sub.sel(height=height)['V']

        ws = cf.wind_uv_to_spd(u, v)
        mws = ws.mean('time')

        # calculate wind power
        power = xr.DataArray(np.interp(ws, power_curve['Wind Speed'], power_curve['Power']), coords=ws.coords)
        meanpower = power.mean('time')
        sdpower = power.std('time')  # power variance

        u_mean = u.mean('time')
        v_mean = v.mean('time')

        # standardize the vectors so they only represent direction
        u_mean_standardize = u_mean / cf.wind_uv_to_spd(u_mean, v_mean)
        v_mean_standardize = v_mean / cf.wind_uv_to_spd(u_mean, v_mean)

        # calculate windspeed variance
        u_variance = np.square(u.std('time'))
        v_variance = np.square(v.std('time'))

        sdwind = np.sqrt(u_variance + v_variance)

        # variance normalized to mean wind speed
        sdwind_norm = sdwind / mws

        plt_vars['meanpower']['data'] = meanpower
        plt_vars['sdpower']['data'] = sdpower
        plt_vars['meanws']['data'] = mws
        plt_vars['sdwind']['data'] = sdwind
        plt_vars['sdwind_norm']['data'] = sdwind_norm

        for pv, plt_info in plt_vars.items():
            for pr, region_info in plt_regions.items():
                if pr == 'windturb':
                    continue
                region_savedir = os.path.join(save_dir, pr)
                os.makedirs(region_savedir, exist_ok=True)

                if 'monthly' in interval_name:
                    sname = '{}_{}_{}m_{}_{}'.format(pr, pv, height, interval_name, t0.strftime('%Y%m%d'))
                    ttl = '{} {}m: {}'.format(plt_info['title'], height, t0.strftime('%b %Y'))
                else:
                    sname = '{}_{}_{}m_{}'.format(pr, pv, height, interval_name)
                    if interval_name == 'seabreeze_days':
                        nm = 'Sea breeze days'
                    elif interval_name == 'noseabreeze_days':
                        nm = 'Non-sea breeze days'
                    elif interval_name == 'seabreeze_morning':
                        # nm = 'Sea breeze days (00Z - 13Z)'
                        nm = 'Sea breeze days (11PM - 8AM EDT)'
                    elif interval_name == 'seabreeze_afternoon':
                        # nm = 'Sea breeze days (14Z - 23Z)'
                        nm = 'Sea breeze days (9AM - 6PM EDT)'
                    elif interval_name == 'noseabreeze_morning':
                        # nm = 'Non-sea breeze days (00Z - 13Z)'
                        nm = 'Non-sea breeze days (11PM - 8AM EDT)'
                    elif interval_name == 'noseabreeze_afternoon':
                        # nm = 'Non-sea breeze days (14Z - 23Z)'
                        nm = 'Non-sea breeze days (9AM - 6PM EDT)'
                    elif interval_name == 'summer2020_all':
                        nm = 'Overall'
                    ttl = '{} {}m\n{}\n{} to {}'.format(plt_info['title'], height, nm, sb_t0str, sb_t1str)
                sfile = os.path.join(region_savedir, sname)

                # set up the map
                lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
                if pv == 'meanws':
                    pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'])
                    quiver_color = 'k'
                    # pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'],
                    #                     ecolor='#777777')
                    # quiver_color = '#777777'
                    # if np.nanmean(plt_info['data']) < mingray['_{}m'.format(height)]:
                    #     pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'],
                    #                         ecolor='gray')
                    #     quiver_color = 'lightgray'
                    # else:
                    #     pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'])
                    #     quiver_color = 'k'
                else:
                    pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'])

                data = plt_info['data']

                # subset grid
                if region_info['subset']:
                    extent = np.add(region_info['extent'], [-.5, .5, -.5, .5]).tolist()
                    data = cf.subset_grid(data, extent)
                    if pv == 'meanws':
                        u_mean_standardize = cf.subset_grid(u_mean_standardize, extent)
                        v_mean_standardize = cf.subset_grid(v_mean_standardize, extent)

                # add lease areas
                if region_info['lease_area']:
                    pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas
                    pf.add_lease_area_polygon(ax, pa_polygon, '#969696')  # planning areas
                    #leasing_areas.plot(ax=ax, lw=.8, color='magenta', transform=ccrs.LambertConformal())

                # add NYSERDA buoy locations
                # nyserda_buoys = cf.nyserda_buoys()
                # for nb, binfo, in nyserda_buoys.items():
                #     ax.plot(binfo['coords']['lon'], binfo['coords']['lat'], c='magenta', mec='k', marker='o', ms=8,
                #             linestyle='none', transform=ccrs.PlateCarree(), zorder=11)
                #     ax.text(binfo['coords']['lon'] + .3, binfo['coords']['lat'], binfo['code'],
                #             bbox=dict(facecolor='lightgray', alpha=0.6), fontsize=7, transform=ccrs.PlateCarree())

                lon = data.XLONG.values
                lat = data.XLAT.values

                # initialize keyword arguments for plotting
                kwargs = dict()

                # plot data
                # pcolormesh: coarser resolution, shows the actual resolution of the model data
                # contourf: smooths the resolution of the model data, plots are less pixelated, can define discrete levels
                if pv == 'meanpower':
                    kwargs['levels'] = list(np.arange(0, 15001, 1000))
                    kwargs['extend'] = 'neither'
                elif pv == 'sdpower':
                    kwargs['levels'] = list(np.arange(2000, 6001, 500))
                    kwargs['extend'] = 'both'
                else:
                    try:
                        vmin = region_info[pv]['limits']['_{}m'.format(height)]['vmin']
                        vmax = region_info[pv]['limits']['_{}m'.format(height)]['vmax']
                        arange_interval = region_info[pv]['limits']['_{}m'.format(height)]['rint']
                        levels = list(np.arange(vmin, vmax + arange_interval, arange_interval))
                        kwargs['levels'] = levels
                    except KeyError:
                        print('no levels specified')

                kwargs['ttl'] = ttl
                kwargs['clab'] = plt_info['color_label']
                pf.plot_contourf(fig, ax, lon, lat, data, plt_info['cmap'], **kwargs)

                # subset the quivers and add as a layer for meanws only
                if pv == 'meanws':
                    if region_info['quiver_subset']:
                        quiver_scale = region_info['quiver_scale']
                        qs = region_info['quiver_subset']['_{}m'.format(height)]
                        ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_mean_standardize.values[::qs, ::qs],
                                  v_mean_standardize.values[::qs, ::qs], scale=quiver_scale, color=quiver_color,
                                  transform=ccrs.PlateCarree())
                    else:
                        ax.quiver(lon, lat, u_mean_standardize.values, v_mean_standardize.values, scale=quiver_scale,
                                  color=quiver_color, transform=ccrs.PlateCarree())

                plt.savefig(sfile, dpi=200)
                plt.close()


def plot_windspeed_differences(ds1, ds2, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250, 200, 160, 10]
    mingray = dict(_10m=5, _160m=5.5, _200m=5.5, _250m=5.5)  # minimum average value for making the state/coastlines and quivers gray

    plt_regions = cf.plot_regions(interval_name)
    plt_vars = dict(meanws_diff=dict(color_label='Average Wind Speed Difference (m/s)',
                                     title='Average Wind Speed Difference',
                                     cmap=plt.get_cmap('RdBu')),
                    meanpower_diff=dict(color_label='Average Estimated 15MW Wind Power Difference (kW)',
                                        title='Average Wind Power Difference (15MW)',
                                        cmap=plt.get_cmap('RdBu')))

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()
    # boem_rootdir = '/Users/garzio/Documents/rucool/bpu/wrf/lease_areas/BOEM_shp_kmls/shapefiles'
    # leasing_areas, planning_areas = cf.boem_shapefiles(boem_rootdir)

    # for calculating power
    power_curve = pd.read_csv('/home/lgarzio/rucool/bpu/wrf/wrf_lw15mw_power.csv')  # on server
    # power_curve = pd.read_csv('/Users/garzio/Documents/rucool/bpu/wrf/wrf_lw15mw_power.csv')

    for height in heights:
        if height == 10:
            u1 = ds1['U10']
            v1 = ds1['V10']
            u2 = ds2['U10']
            v2 = ds2['V10']
        else:
            u1 = ds1.sel(height=height)['U']
            v1 = ds1.sel(height=height)['V']
            u2 = ds2.sel(height=height)['U']
            v2 = ds2.sel(height=height)['V']

        # calculate windspeed difference
        ws1 = cf.wind_uv_to_spd(u1, v1)
        ws2 = cf.wind_uv_to_spd(u2, v2)
        mws1 = ws1.mean('time')
        mws2 = ws2.mean('time')
        mws_diff = mws1 - mws2

        # calculate power difference
        power1 = xr.DataArray(np.interp(ws1, power_curve['Wind Speed'], power_curve['Power']), coords=ws1.coords)
        power2 = xr.DataArray(np.interp(ws2, power_curve['Wind Speed'], power_curve['Power']), coords=ws2.coords)
        meanpower1 = power1.mean('time')
        meanpower2 = power2.mean('time')
        power_diff = meanpower1 - meanpower2

        plt_vars['meanws_diff']['data'] = mws_diff
        plt_vars['meanpower_diff']['data'] = power_diff

        for pv, plt_info in plt_vars.items():
            for pr, region_info in plt_regions.items():
                if pr == 'windturb':
                    continue
                region_savedir = os.path.join(save_dir, pr)
                os.makedirs(region_savedir, exist_ok=True)

                if 'monthly' in interval_name:
                    sname = '{}_{}_{}m_{}_{}'.format(pr, pv, height, interval_name, t0.strftime('%Y%m%d'))
                    ttl = '{} {}m: {}'.format(plt_info['title'], height, t0.strftime('%b %Y'))
                else:
                    sname = '{}_{}_{}m_{}'.format(pr, pv, height, interval_name)
                    if interval_name == 'diff_morning':
                        nm = 'Sea breeze minus Non-sea breeze (00Z - 13Z)'
                    elif interval_name == 'diff_afternoon':
                        nm = 'Sea breeze minus Non-sea breeze (14Z - 23Z)'
                    elif interval_name == 'diff_seabreeze':
                        nm = 'Sea breeze days: Afternoon minus Morning'
                    elif interval_name == 'diff_noseabreeze':
                        nm = 'Non-sea breeze days: Afternoon minus Morning'
                    ttl = '{} {}m\n{}\n{} to {}'.format(plt_info['title'], height, nm, sb_t0str, sb_t1str)
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

                # add NYSERDA buoy locations
                # nyserda_buoys = cf.nyserda_buoys()
                # for nb, binfo, in nyserda_buoys.items():
                #     ax.plot(binfo['coords']['lon'], binfo['coords']['lat'], c='magenta', mec='k', marker='o', ms=8,
                #             linestyle='none', transform=ccrs.PlateCarree(), zorder=11)
                #     ax.text(binfo['coords']['lon'] + .3, binfo['coords']['lat'], binfo['code'],
                #             bbox=dict(facecolor='lightgray', alpha=0.6), fontsize=7, transform=ccrs.PlateCarree())

                lon = data.XLONG.values
                lat = data.XLAT.values

                # initialize keyword arguments for plotting
                kwargs = dict()

                # plot data
                # pcolormesh: coarser resolution, shows the actual resolution of the model data
                # contourf: smooths the resolution of the model data, plots are less pixelated, can define discrete levels
                if pv == 'meanpower_diff':
                    kwargs['levels'] = list(np.arange(-4000, 4001, 1000))
                else:
                    try:
                        vmin = region_info[pv]['limits']['_{}m'.format(height)]['vmin']
                        vmax = region_info[pv]['limits']['_{}m'.format(height)]['vmax']
                        arange_interval = region_info[pv]['limits']['_{}m'.format(height)]['rint']
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

    # break up dates into the plotting interval specified
    if 'seabreeze' in intvl:
        ds = ds.sel(time=slice(sdate, edate))
        dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
        dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df_sb = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df_sb['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # define arguments for plotting function
        kwargs = dict()
        kwargs['sb_t0str'] = dst0
        kwargs['sb_t1str'] = dst1

        # grab the WRF data for the seabreeze dates
        ds_sb = ds.sel(time=sb_datetimes)
        # ds_sb = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 15, 0)))  # for debugging

        # grab the WRF data for the non-seabreeze dates
        df_nosb = df[df['Seabreeze'] == 'n']
        nosb_dates = np.array(pd.to_datetime(df_nosb['Date']))
        nosb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x
                          in nosb_dates]
        nosb_datetimes = pd.to_datetime(sorted([inner for outer in nosb_datetimes for inner in outer]))
        ds_nosb = ds.sel(time=nosb_datetimes)
        # ds_nosb = ds.sel(time=slice(dt.datetime(2020, 6, 2, 0, 0), dt.datetime(2020, 6, 2, 15, 0)))  # for debugging

        if intvl == 'seabreeze_days':
            plot_averages(ds_sb, savedir, 'seabreeze_days', **kwargs)
            plot_averages(ds_nosb, savedir, 'noseabreeze_days', **kwargs)
        else:
            hours_sb = pd.to_datetime(ds_sb.time.values).hour
            ds_sb_morn = ds_sb.isel(time=np.logical_and(hours_sb >= 3, hours_sb < 13))  # subset morning hours
            ds_sb_aft = ds_sb.isel(time=np.logical_and(hours_sb >= 13, hours_sb < 23))  # subset afternoon hours

            hours_nosb = pd.to_datetime(ds_nosb.time.values).hour
            ds_nosb_morn = ds_nosb.isel(time=np.logical_and(hours_nosb >= 3, hours_nosb < 13))  # subset morning hours
            ds_nosb_aft = ds_nosb.isel(time=np.logical_and(hours_nosb >= 13, hours_nosb < 23))  # subset afternoon hours

            if intvl == 'seabreeze_hours':
                plot_averages(ds_sb_morn, savedir, 'seabreeze_morning', **kwargs)
                plot_averages(ds_sb_aft, savedir, 'seabreeze_afternoon', **kwargs)
                plot_averages(ds_nosb_morn, savedir, 'noseabreeze_morning', **kwargs)
                plot_averages(ds_nosb_aft, savedir, 'noseabreeze_afternoon', **kwargs)
            elif intvl == 'seabreeze_diff':
                # difference between seabreeze and non-seabreeze morning/afternoon
                plot_windspeed_differences(ds_sb_morn, ds_nosb_morn, savedir, 'diff_morning', **kwargs)
                plot_windspeed_differences(ds_sb_aft, ds_nosb_aft, savedir, 'diff_afternoon', **kwargs)

                # difference between morning and afternoon on seabreeze and non-seabreeze days
                plot_windspeed_differences(ds_sb_aft, ds_sb_morn, savedir, 'diff_seabreeze', **kwargs)
                plot_windspeed_differences(ds_nosb_aft, ds_nosb_morn, savedir, 'diff_noseabreeze', **kwargs)
    elif 'all' in intvl:
        ds = ds.sel(time=slice(sdate, edate))
        # ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 5, 0)))  # for debugging
        dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
        dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')
        kwargs = dict()
        kwargs['sb_t0str'] = dst0
        kwargs['sb_t1str'] = dst1
        plot_averages(ds, savedir, intvl, **kwargs)

    else:
        start, end = cf.daterange_interval(intvl, sdate, edate)

        # grab the WRF data for each interval
        for sd, ed in zip(start, end):
            # sd = dt.datetime(2019, 9, 1, 0, 0)  # for debugging
            # ed = dt.datetime(2019, 9, 1, 5, 0)  # for debugging
            # dst = ds.sel(time=slice(sd, ed))  # for debugging
            dst = ds.sel(time=slice(sd, ed + dt.timedelta(hours=23)))
            kwargs = dict()
            kwargs['t0'] = pd.to_datetime(dst.time.values[0])
            plot_averages(dst, savedir, intvl, **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 8, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'seabreeze_hours'  # 'monthly' 'seabreeze_days' 'seabreeze_hours' 'seabreeze_diff' 'summer2020_all'
    main(save_directory, start_date, end_date, interval)
