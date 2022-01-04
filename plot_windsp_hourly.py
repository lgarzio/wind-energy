#!/usr/bin/env python

"""
Author: Lori Garzio on 10/5/2021
Last modified: 1/4/2022
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


def plot_windspeed_power(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250, 200, 160, 10]

    plt_regions = cf.plot_regions(interval_name)
    plt_vars = dict(meanws=dict(color_label='Wind Speed (m/s)',
                                title='Wind Speed',
                                cmap=plt.get_cmap('BuPu')),
                    meanpower=dict(color_label='Estimated 15MW Wind Power (kW)',
                                   title='Wind Power (15MW)',
                                   cmap='OrRd')
                    )

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

        hours = np.arange(1, 24)

        # plot each hour
        for hour in hours:
            u_hour = u[u.time.dt.hour == hour]
            v_hour = v[v.time.dt.hour == hour]
            ws_hour = cf.wind_uv_to_spd(u_hour, v_hour)

            # standardize the vectors so they only represent direction
            u_hour_standardize = u_hour / cf.wind_uv_to_spd(u_hour, v_hour)
            v_hour_standardize = v_hour / cf.wind_uv_to_spd(u_hour, v_hour)

            # calculate wind power
            power = xr.DataArray(np.interp(ws_hour, power_curve['Wind Speed'], power_curve['Power']), coords=ws_hour.coords)

            plt_vars['meanws']['data'] = ws_hour
            plt_vars['meanpower']['data'] = power

            for pv, plt_info in plt_vars.items():
                for pr, region_info in plt_regions.items():
                    region_savedir = os.path.join(save_dir, pr)
                    os.makedirs(region_savedir, exist_ok=True)

                    sname = '{}_{}_{}m_{}_H{}'.format(pr, pv, height, interval_name, str(hour).zfill(3))
                    if interval_name == 'hourly_avg':
                        ttl = '{} {}m: H{}\n{} to {}'.format(plt_info['title'], height, str(hour).zfill(3),
                                                             sb_t0str, sb_t1str)
                    elif interval_name == 'hourly_cases':
                        ttl = '{} {}m: H{}\n{} to {}'.format(plt_info['title'], height, str(hour).zfill(3),
                                                             sb_t0str, sb_t1str)
                    else:
                        if interval_name == 'seabreeze_days_hourly_avg':
                            nm = 'Sea breeze days'
                        elif interval_name == 'noseabreeze_days_hourly_avg':
                            nm = 'Non-sea breeze days'
                        ttl = '{} {}m\n{}: H{}\n{} to {}'.format(plt_info['title'], height, nm, str(hour).zfill(3),
                                                                 sb_t0str, sb_t1str)

                    sfile = os.path.join(region_savedir, sname)

                    # set up the map
                    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
                    if pv == 'meanws':
                        pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'])
                        quiver_color = 'k'
                    else:
                        pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'])

                    data = plt_info['data']

                    # subset grid
                    if region_info['subset']:
                        extent = np.add(region_info['extent'], [-.5, .5, -.5, .5]).tolist()
                        data = cf.subset_grid(data, extent)
                        if pv == 'meanws':
                            u_hour_standardize = cf.subset_grid(u_hour_standardize, extent)
                            v_hour_standardize = cf.subset_grid(v_hour_standardize, extent)

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
                            ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_hour_standardize.values[::qs, ::qs],
                                      v_hour_standardize.values[::qs, ::qs], scale=quiver_scale,
                                      color=quiver_color, transform=ccrs.PlateCarree())
                        else:
                            ax.quiver(lon, lat, u_hour_standardize.values, v_hour_standardize.values,
                                      scale=quiver_scale, color=quiver_color, transform=ccrs.PlateCarree())

                    plt.savefig(sfile, dpi=200)
                    plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    if intvl == 'hourly_cases':
        savedir = os.path.join(sDir, 'seabreeze_cases_surface_maps', '{}_{}'.format(intvl, sdate.strftime('%Y%m%d')))
    else:
        savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)

    ds = ds.sel(time=slice(sdate, edate))
    # ds = ds.sel(time=slice(dt.datetime(2020, 6, 8, 0, 0), dt.datetime(2020, 6, 8, 5, 0)))  # for debugging
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1
    plot_windspeed_power(ds, savedir, intvl, **kwargs)


if __name__ == '__main__':
    #save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 8, 0, 0)  # dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 6, 8, 23, 0)  # dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'hourly_cases'  # 'hourly_cases'
    main(save_directory, start_date, end_date, interval)
