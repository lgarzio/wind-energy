#!/usr/bin/env python

"""
Author: Lori Garzio on 10/28/2021
Last modified: 10/28/2021
Plot divergence of hourly-averaged wind speeds at 500m from the native model level files
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
import metpy.calc as mc
from wrf import interplevel, default_fill
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plot_divergence(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [500]

    plt_regions = cf.plot_regions(interval_name)
    plt_vars = dict(divergence=dict(color_label='Divergence x $10^{-4}$ (1/s)',
                                    title='Divergence',
                                    cmap=plt.get_cmap('RdBu_r')))

    la_polygon, pa_polygon = cf.extract_lease_area_outlines()

    for height in heights:
        u = interplevel(ds_sub.u, ds_sub.height_agl, height, default_fill(np.float32))
        v = interplevel(ds_sub.v, ds_sub.height_agl, height, default_fill(np.float32))

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

            div = mc.divergence(uhm, vhm)
            plt_vars['divergence']['data'] = div * 10**4  # surface divergence, *10^-4 1/s

            for pv, plt_info in plt_vars.items():
                for pr, region_info in plt_regions.items():
                    region_savedir = os.path.join(save_dir, pr)
                    os.makedirs(region_savedir, exist_ok=True)

                    sname = '{}_{}m_{}_H{}'.format(pr, height, interval_name, str(hour).zfill(3))
                    if interval_name == 'divergence_seabreeze_days_hourly_avg':
                        ttl = 'Sea Breeze Days\nHourly Averaged {} {}m: H{}\n{} to {}'.format(plt_info['title'], height,
                                                                                              str(hour).zfill(3),
                                                                                              sb_t0str, sb_t1str)
                    else:
                        ttl = 'Sea Breeze Days\n{} {}m: H{}\n{}'.format(plt_info['title'], height, str(hour).zfill(3),
                                                                        sb_t0str)
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

                    lon = data.XLONG.values
                    lat = data.XLAT.values

                    # initialize keyword arguments for plotting
                    kwargs = dict()

                    # plot data
                    if interval_name == 'divergence_seabreeze_days_hourly_avg':
                        kwargs['levels'] = [-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5,
                                            0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
                        kwargs['cbar_ticks'] = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
                    else:
                        kwargs['levels'] = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5,
                                            3, 3.5, 4, 4.5, 5]
                        kwargs['cbar_ticks'] = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
                    kwargs['extend'] = 'both'

                    kwargs['ttl'] = ttl
                    kwargs['clab'] = plt_info['color_label']
                    kwargs['shift_subplot_right'] = 0.85
                    pf.plot_contourf(fig, ax, lon, lat, data, plt_info['cmap'], **kwargs)

                    # subset the quivers and add as a layer
                    if region_info['quiver_subset']:
                        quiver_scale = region_info['quiver_scale']
                        qs = region_info['quiver_subset']['_{}m'.format(height)]
                        ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_hourly_mean_standardize.values[::qs, ::qs],
                                  v_hourly_mean_standardize.values[::qs, ::qs], scale=quiver_scale,
                                  color=quiver_color, transform=ccrs.PlateCarree())
                    else:
                        ax.quiver(lon, lat, u_hourly_mean_standardize.values, v_hourly_mean_standardize.values,
                                  scale=quiver_scale, color=quiver_color, transform=ccrs.PlateCarree())

                    plt.savefig(sfile, dpi=200)
                    plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'https://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_native_levels/WRF_4.1_3km_Native_Levels_Dataset_Best'

    savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')
    if intvl == 'divergence_seabreeze_days_hourly_avg':
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # grab the WRF data for the seabreeze dates
        ds = ds.sel(time=sb_datetimes)
        # ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 5, 0)))  # for debugging

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    plot_divergence(ds, savedir, intvl, **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'divergence_seabreeze_days_hourly_avg'  # divergence_seabreeze_days_hourly_avg  divergence_hourly
    main(save_directory, start_date, end_date, interval)
