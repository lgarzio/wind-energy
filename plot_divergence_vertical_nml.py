#!/usr/bin/env python

"""
Author: Lori Garzio on 11/17/2021
Last modified: 12/9/2021
Plot horizontal slices of divergence of hourly-averaged wind speeds from the native model level files
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import functions.plotting as pf
import metpy.calc as mc
from wrf import interplevel, default_fill, interpline, CoordPair, WrfProj
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified
np.set_printoptions(suppress=True)


def plot_divergence_horizontal(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = np.arange(100, 2100, 100)

    #bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    bathymetry = '/home/lgarzio/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'  # on server
    extent = [-78, -70, 37, 41]  # subset the file so it's easier to work with
    bathy = xr.open_dataset(bathymetry)
    bathy = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                      lat=slice(extent[2] - .1, extent[3] + .1))

    lon = ds_sub.XLONG.values
    lat = ds_sub.XLAT.values

    #lm = ds_sub.LANDMASK.mean('time')

    # grab data along the line perpendicular to the coast in southern NJ
    point_start = CoordPair(lat=40.7, lon=-76)  # for the line perpendicular to the coast
    point_end = CoordPair(lat=38, lon=-72.8)  # for the line perpendicular to the coast

    u = interplevel(ds_sub.u, ds_sub.height_agl, heights, default_fill(np.float32))
    v = interplevel(ds_sub.v, ds_sub.height_agl, heights, default_fill(np.float32))

    hours = np.arange(1, 24)

    for hour in hours:
        u_hour = u[u.time.dt.hour == hour]
        v_hour = v[v.time.dt.hour == hour]

        # calculate hourly average
        uhm = u_hour.mean('time')
        vhm = v_hour.mean('time')

        div = mc.divergence(uhm, vhm) * 10**4  # surface divergence, *10^-4 1/s
        # get values along specified line
        wrf_projection = WrfProj(map_proj=1, dx=3000, dy=3000, truelat1=38.8, truelat2=38.8,
                                 moad_cen_lat=38.75293, stand_lon=-74.5)
        ll_point = CoordPair(lat=34.321144, lon=-79.763733)
        div_line = interpline(div, start_point=point_start, end_point=point_end, projection=wrf_projection,
                              ll_point=ll_point, latlon=True)

        # get the coordinates for the line that is returned
        if hour == 1:
            # get the bathymetry along the interpolated line

            lats_interp = np.array([])
            lons_interp = np.array([])
            land_mask = np.array([])
            plot_elev = np.array([])
            for i, value in enumerate(div_line.xy_loc.values):
                lats_interp = np.append(lats_interp, value.lat)
                lons_interp = np.append(lons_interp, value.lon)

                # # find the land mask at the closest grid point
                # # calculate the sum of the absolute value distance between the model location and buoy location
                # a = abs(lat - value.lat) + abs(lon - value.lon)
                #
                # # find the indices of the minimum value in the array calculated above
                # i, j = np.unravel_index(a.argmin(), a.shape)
                # land_mask = np.append(land_mask, lm[i, j].values)

                # find the bathymetry at the closest point
                minlat_idx = np.argmin(abs(bathy.lat.values - value.lat))
                minlon_idx = np.argmin(abs(bathy.lon.values - value.lon))
                plot_elev = np.append(plot_elev, bathy.elevation[minlat_idx, minlon_idx])

            # find the edge of the continental shelf
            elev_mask = plot_elev < -1000
            elev_idx = np.where(elev_mask[:-1] != elev_mask[1:])[0]

        if interval_name == 'divergence_seabreeze_days_hourly_avg':
            ttl = 'Cross-section of Horizontal Divergence: H{}\nSea Breeze Days\n{} to {}'.format(str(hour).zfill(3),
                                                                                                   sb_t0str, sb_t1str)
        elif interval_name == 'divergence_nonseabreeze_days_hourly_avg':
            ttl = 'Cross-section of Horizontal Divergence: H{}\nNon-Sea Breeze Days\n{} to {}'.format(str(hour).zfill(3),
                                                                                                   sb_t0str, sb_t1str)
        elif interval_name == 'divergence_hourly_cases_horizontal':
            ttl = 'Cross-section of Horizontal Divergence\n{} H{}'.format(sb_t0str, str(hour).zfill(3))

        if 'cases' in interval_name:
            levels = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            ticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        else:
            levels = [-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
                      0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
            ticks = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]

        fig, ax = plt.subplots(figsize=(9, 8))

        # initialize keyword arguments for plotting
        kwargs = dict()
        kwargs['cbar_ticks'] = ticks
        cmap = plt.get_cmap('RdBu_r')  # for pcolormesh only
        kwargs['cmap'] = cmap  # for pcolormesh only
        levels = levels  # for pcolormesh only
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)  # for pcolormesh only
        kwargs['norm_clevs'] = norm  # for pcolormesh only

        kwargs['ttl'] = ttl
        kwargs['clab'] = 'Divergence x $10^{-4}$ (1/s)'
        #kwargs['shift_subplot_right'] = 0.97
        # kwargs['xlab'] = 'Distance From Shore (km)'
        kwargs['xlab'] = 'Longitude'
        kwargs['ylab'] = 'Height (m)'
        #kwargs['yticks'] = [5, 10, 15, 20]
        pf.plot_pcolormesh(fig, ax, lons_interp, heights, div_line.values, **kwargs)

        ylims = ax.get_ylim()
        # # add a line for the coast
        coastline_lon = -74.40
        ax.vlines(coastline_lon, ylims[0], ylims[1], colors='k', ls='-')
        # ax.vlines(0, ylims[0], ylims[1], colors='k', ls='-')

        # add lines for the wind energy area (calculated in hovmoller_line_map.py)
        # wea1 = 14.73  # for distance from shore
        # wea2 = 36.32  # for distance from shore
        wea1 = -74.28  # for longitude
        wea2 = -74.12  # for longitude
        ax.vlines(wea1, ylims[0], ylims[1], colors='darkgray', ls='--')
        ax.vlines(wea2, ylims[0], ylims[1], colors='darkgray', ls='--')

        # add a dot at the shelf break
        ax.plot(lons_interp[elev_idx], ylims[0] * 1.2, 'ko')

        ax.set_ylim(ylims)
        # ax.set_xlim([-200, 200])

        sname = '{}_horizontal_H{}.png'.format(interval_name.split('_hourly_avg')[0], str(hour).zfill(3))
        plt.savefig(os.path.join(save_dir, sname), dpi=200)
        plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'https://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_native_levels/WRF_4.1_3km_Native_Levels_Dataset_Best'

    if intvl == 'divergence_hourly_cases_horizontal':
        savedir = os.path.join(sDir, 'hovmoller_seabreeze_cases_horizontal', '{}_{}'.format(intvl, sdate.strftime('%Y%m%d')))
    else:
        savedir = os.path.join(sDir, '{}_{}-{}-horizontal'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    if intvl == 'divergence_seabreeze_days_hourly_avg':
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # grab the WRF data for the seabreeze dates
        ds = ds.sel(time=sb_datetimes)
        # ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 5, 0)))  # for debugging
        plot_divergence_horizontal(ds, savedir, intvl, **kwargs)

    elif intvl == 'divergence_nonseabreeze_days_hourly_avg':
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # grab the WRF data for the non-seabreeze dates
        nosb_datetimes = [t for t in ds.time.values if t not in sb_datetimes]
        ds_nosb = ds.sel(time=nosb_datetimes)
        # ds_nosb = ds.sel(time=slice(dt.datetime(2020, 6, 2, 0, 0), dt.datetime(2020, 6, 2, 5, 0)))  # for debugging
        plot_divergence_horizontal(ds_nosb, savedir, intvl, **kwargs)

    else:
        # ds = ds.sel(time=slice(dt.datetime(2020, 6, 8, 0, 0), dt.datetime(2020, 6, 8, 2, 0)))  # for debugging
        plot_divergence_horizontal(ds, savedir, intvl, **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 8, 0, 0)  # dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 6, 8, 23, 0)  # dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'divergence_hourly_cases_horizontal'  # divergence_seabreeze_days_hourly_avg divergence_nonseabreeze_days_hourly_avg  divergence_hourly_cases_horizontal - use this for daily intervals
    main(save_directory, start_date, end_date, interval)
