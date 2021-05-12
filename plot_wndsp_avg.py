#!/usr/bin/env python

"""
Author: Lori Garzio on 4/12/2021
Last modified: 5/11/2021
Plot average WRF windspeeds at 10m and 160m at user-defined grouping intervals
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'
    heights = [160, 10]
    mingray = dict(_10m=5, _160m=7)  # minimum average value for making the state/coastlines and quivers gray

    plt_regions = cf.plot_regions()
    plt_vars = dict(meanws=dict(color_label='Average Wind Speed (m/s)',
                                title='Average Wind Speed',
                                cmap=plt.get_cmap('viridis')),
                    sdwind=dict(color_label='Variance (m/s)',
                                title='Wind Speed Variance',
                                cmap='BuPu'),
                    sdwind_norm=dict(color_label='Normalized Variance',
                                     title='Normalized Wind Speed Variance',
                                     cmap='BuPu'))

    la_polygon = cf.extract_lease_area_outlines()

    savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    # break up date range into the plotting interval specified
    start, end = cf.daterange_interval(intvl, sdate, edate)

    ds = xr.open_dataset(wrf)
    # grab the WRF data for each interval
    for sd, ed in zip(start, end):
        # sd = dt.datetime(2019, 9, 1, 0, 0)  # for debugging
        # ed = dt.datetime(2019, 9, 1, 5, 0)  # for debugging
        # dst = ds.sel(time=slice(sd, ed))  # for debugging
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

            # calculate the variance
            u_variance = np.square(u.std('time'))
            v_variance = np.square(v.std('time'))

            sdwind = np.sqrt(u_variance + v_variance)

            # variance normalized to mean wind speed
            sdwind_norm = sdwind / mws

            plt_vars['meanws']['data'] = mws
            plt_vars['sdwind']['data'] = sdwind
            plt_vars['sdwind_norm']['data'] = sdwind_norm

            for pv, plt_info in plt_vars.items():
                for pr, region_info in plt_regions.items():
                    region_savedir = os.path.join(savedir, pr)
                    os.makedirs(region_savedir, exist_ok=True)

                    sname = '{}_{}_{}m_{}_{}'.format(pr, pv, height, intvl, sd.strftime('%Y%m%d'))
                    sfile = os.path.join(region_savedir, sname)
                    #sfile = os.path.join(savedir, sname)  # for debugging

                    # set up the map
                    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
                    if pv == 'meanws':
                        if np.nanmean(plt_info['data']) < mingray['_{}m'.format(height)]:
                            pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'],
                                                ecolor='gray')
                            quiver_color = 'lightgray'
                        else:
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
                            u_mean_standardize = cf.subset_grid(u_mean_standardize, extent)
                            v_mean_standardize = cf.subset_grid(v_mean_standardize, extent)

                    # add lease areas
                    if region_info['lease_area']:
                        pf.add_lease_area_polygon(ax, la_polygon, 'magenta')

                    lon = data.XLONG.values
                    lat = data.XLAT.values

                    # initialize keyword arguments for plotting
                    kwargs = dict()

                    # plot data
                    # pcolormesh: coarser resolution, shows the actual resolution of the model data
                    # contourf: smooths the resolution of the model data, plots are less pixelated, can define discrete levels
                    try:
                        vmin = region_info[pv]['limits']['_{}m'.format(height)]['vmin']
                        vmax = region_info[pv]['limits']['_{}m'.format(height)]['vmax']
                        levels = list(np.arange(vmin, vmax + .5, .5))
                    except KeyError:
                        levels = list(np.arange(.9, 1.3, .05))  # for normalized variance plots
                    ttl = '{} {}m: {}'.format(plt_info['title'], height, sd.strftime('%b %Y'))
                    kwargs['levels'] = levels
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


if __name__ == '__main__':
    #save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_dir = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 9, 1, 0, 0)
    interval = 'monthly'
    main(save_dir, start_date, end_date, interval)
