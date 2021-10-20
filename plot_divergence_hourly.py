#!/usr/bin/env python

"""
Author: Lori Garzio on 10/19/2021
Last modified: 10/20/2021
Plot divergence of hourly-averaged wind speeds
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plot_divergence(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250, 200, 160, 10]

    plt_regions = cf.plot_regions(interval_name)
    plt_vars = dict(divergence=dict(color_label='Divergence (1/s)',
                                    title='Divergence',
                                    cmap=plt.get_cmap('RdBu')))

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    lon = ds_sub.XLONG.values
    lat = ds_sub.XLAT.values

    for height in heights:
        if height == 10:
            u = ds_sub['U10']
            v = ds_sub['V10']
        else:
            u = ds_sub.sel(height=height)['U']
            v = ds_sub.sel(height=height)['V']

        hours = np.arange(1, 24)

        for hour in hours:
            u_hour = u[u.time.dt.hour == hour]
            v_hour = v[v.time.dt.hour == hour]

            # calculate hourly average
            uhm = u_hour.mean('time')
            vhm = v_hour.mean('time')

            # standardize the vectors so they only represent direction
            u_hourly_mean_standardize = uhm / cf.wind_uv_to_spd(uhm, vhm)
            v_hourly_mean_standardize = vhm / cf.wind_uv_to_spd(uhm, vhm)

            # calculate divergence - translated from Laura Nazzaro's MATLAB code
            div = np.empty(np.shape(uhm.values))
            div[:] = np.nan

            for i in range(len(uhm.values)):
                for j in range(len(vhm.values)):
                    if np.logical_and(i > 0, j > 0):  # edges are nan
                        if np.logical_and(i < len(uhm.values) - 1, j < len(vhm.values) - 1):  # edges are nan
                            vy1 = vhm.values[i, j + 1]  # v in y direction from center point
                            vy2 = vhm.values[i, j - 1]  # v in opposite y direction from center point
                            ux1 = uhm.values[i + 1, j]  # u in x direction from center point
                            ux2 = uhm.values[i - 1, j]  # u in opposite x direction from center point

                            # calculate distances between points
                            geod = Geodesic.WGS84
                            g = geod.Inverse(lat[i, j + 1], lon[i, j + 1], lat[i, j - 1], lon[i, j - 1])
                            dy1minus2_meters = np.round(g['s12'])
                            g = geod.Inverse(lat[i + 1, j], lon[i + 1, j], lat[i - 1, j], lon[i - 1, j])
                            dx1minus2_meters = np.round(g['s12'])

                            # calculate dudx, dvdx, dudy, dvdy at one gridpoint distance from center point
                            dudx = (ux1 - ux2) / dx1minus2_meters
                            dvdy = (vy1 - vy2) / dy1minus2_meters

                            div[i, j] = dudx + dvdy  # surface divergence in m/s

            plt_vars['divergence']['data'] = xr.DataArray(div, coords=uhm.coords, dims=uhm.dims)

            for pv, plt_info in plt_vars.items():
                for pr, region_info in plt_regions.items():
                    region_savedir = os.path.join(save_dir, pr)
                    os.makedirs(region_savedir, exist_ok=True)

                    sname = '{}_{}m_{}_H{}'.format(pr, height, interval_name, str(hour).zfill(3))
                    ttl = 'Sea Breeze Days\nHourly Averaged {} {}m: H{}\n{} to {}'.format(plt_info['title'], height,
                                                                                          str(hour).zfill(3),
                                                                                          sb_t0str, sb_t1str)
                    sfile = os.path.join(region_savedir, sname)

                    # set up the map
                    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
                    pf.add_map_features(ax, region_info['extent'], region_info['xticks'], region_info['yticks'])
                    quiver_color = 'k'

                    data = plt_info['data']

                    # subset grid
                    if region_info['subset']:
                        extent = np.add(region_info['extent'], [-.5, .5, -.5, .5]).tolist()
                        data = cf.subset_grid(data, extent)
                        u_hourly_mean_standardize = cf.subset_grid(u_hourly_mean_standardize, extent)
                        v_hourly_mean_standardize = cf.subset_grid(v_hourly_mean_standardize, extent)

                    # add lease areas
                    if region_info['lease_area']:
                        pf.add_lease_area_polygon(ax, la_polygon, '#737373')  # lease areas
                        pf.add_lease_area_polygon(ax, pa_polygon, '#969696')  # planning areas
                        #leasing_areas.plot(ax=ax, lw=.8, color='magenta', transform=ccrs.LambertConformal())

                    lon_plot = data.XLONG.values
                    lat_plot = data.XLAT.values

                    # initialize keyword arguments for plotting
                    kwargs = dict()

                    # plot data
                    # pcolormesh: coarser resolution, shows the actual resolution of the model data
                    # contourf: smooths the resolution of the model data, plots are less pixelated, can define discrete levels
                    kwargs['levels'] = [-.0004, -.0003, -.0002, -.0001, 0, .0001, .0002, .0003, .0004]
                    kwargs['extend'] = 'both'

                    kwargs['ttl'] = ttl
                    kwargs['clab'] = plt_info['color_label']
                    kwargs['shift_subplot_right'] = 0.85
                    pf.plot_contourf(fig, ax, lon_plot, lat_plot, data, plt_info['cmap'], **kwargs)

                    # subset the quivers and add as a layer
                    if region_info['quiver_subset']:
                        quiver_scale = region_info['quiver_scale']
                        qs = region_info['quiver_subset']['_{}m'.format(height)]
                        ax.quiver(lon_plot[::qs, ::qs], lat_plot[::qs, ::qs], u_hourly_mean_standardize.values[::qs, ::qs],
                                  v_hourly_mean_standardize.values[::qs, ::qs], scale=quiver_scale,
                                  color=quiver_color, transform=ccrs.PlateCarree())
                    else:
                        ax.quiver(lon_plot, lat_plot, u_hourly_mean_standardize.values, v_hourly_mean_standardize.values,
                                  scale=quiver_scale, color=quiver_color, transform=ccrs.PlateCarree())

                    plt.savefig(sfile, dpi=200)
                    plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))
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
    # ds_sb = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 5, 0)))  # for debugging

    plot_divergence(ds_sb, savedir, 'seabreeze_days_hourly_avg_divergence', **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'divergence'
    main(save_directory, start_date, end_date, interval)
