#!/usr/bin/env python

"""
Author: Lori Garzio on 1/25/2022
Last modified: 1/25/2022
Plot Hovmoller diagrams of 2m Air Temperature at specified cross-section
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cmocean as cmo
import functions.plotting as pf
from wrf import interpline, CoordPair, WrfProj
from geographiclib.geodesic import Geodesic
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plot_airtemp_hovmoller(ds_sub, save_dir, interval_name, line, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None

    lon = ds_sub.XLONG.values
    lat = ds_sub.XLAT.values

    lm = ds_sub.LANDMASK.mean('time')

    if line == 'short_perpendicular':
        # grab data along the line perpendicular to the coast in southern NJ
        point_start = CoordPair(lat=39.9, lon=-75)  # for the line perpendicular to the coast (short line)
        point_end = CoordPair(lat=38.87, lon=-73.8)  # for the line perpendicular to the coast (short line)
        div_shape = 52
    elif line == 'long_perpendicular':
        point_start = CoordPair(lat=40.7, lon=-76)  # for the line perpendicular to the coast (long line)
        point_end = CoordPair(lat=38, lon=-72.8)  # for the line perpendicular to the coast  (long line)
        div_shape = 136
    elif line == 'wea':
        point_start = CoordPair(lat=38.7, lon=-74.8)  # for the line parallel to the coast (along the WEA)
        point_end = CoordPair(lat=39.7, lon=-73.5)  # for the line parallel to the coast (along the WEA)
        div_shape = 53

    t2 = ds_sub.T2 - 273.15  # convert K to C

    hours = np.arange(13, 23)

    t2_final = np.empty(shape=(len(hours), div_shape))

    t2_final[:] = np.nan

    for hour_idx, hour in enumerate(hours):
        t2_hour = t2[t2.time.dt.hour == hour]
        mean_t2 = t2_hour.mean('time')

        # get values along specified line
        wrf_projection = WrfProj(map_proj=1, dx=3000, dy=3000, truelat1=38.8, truelat2=38.8,
                                 moad_cen_lat=38.75293, stand_lon=-74.5)
        ll_point = CoordPair(lat=34.321144, lon=-79.763733)
        mean_t2_line = interpline(mean_t2, start_point=point_start, end_point=point_end, projection=wrf_projection,
                                  ll_point=ll_point, latlon=True)

        # get the coordinates for the line that is returned
        if hour_idx == 0:
            # get the bathymetry along the interpolated line
            lats_interp = np.array([])
            lons_interp = np.array([])
            land_mask = np.array([])
            for i, value in enumerate(mean_t2_line.xy_loc.values):
                lats_interp = np.append(lats_interp, value.lat)
                lons_interp = np.append(lons_interp, value.lon)

                # find the land mask at the closest grid point
                # calculate the sum of the absolute value distance between the model location and buoy location
                a = abs(lat - value.lat) + abs(lon - value.lon)

                # find the indices of the minimum value in the array calculated above
                i, j = np.unravel_index(a.argmin(), a.shape)
                land_mask = np.append(land_mask, lm[i, j].values)

            if '_perpendicular' in line:
                # find the coastline longitude (where landmask values change from 1 to 0)
                # have to take the max because the line crosses Delaware River
                coastline_idx = [np.nanmax(np.where(land_mask[:-1] != land_mask[1:])[0])]
                coastline_lon = np.mean(lons_interp[coastline_idx[0]:coastline_idx[0] + 2])
                coastline_lat = np.mean(lats_interp[coastline_idx[0]:coastline_idx[0] + 2])

                # calculate the distance from each coordinate to the coastline
                # negative values are land-side, positive values are ocean-side
                distance_km = np.array([])
                geod = Geodesic.WGS84
                for i, lati in enumerate(lats_interp):
                    g = geod.Inverse(coastline_lat, coastline_lon, lati, lons_interp[i])
                    dist_km = g['s12'] * .001
                    if i <= coastline_idx[0]:
                        dist_km = -dist_km
                    distance_km = np.append(distance_km, dist_km)

        t2_final[hour_idx] = mean_t2_line

    if interval_name == 'airtemp_hourly_avg_hovmoller_zoomed_seabreeze':
        ttl = 'Hourly Averaged Seabreeze Days\n2m Air Temperature (\N{DEGREE SIGN}C)\n{} to {}'.format(sb_t0str, sb_t1str)

    elif interval_name == 'airtemp_hourly_avg_hovmoller_zoomed_noseabreeze':
        ttl = 'Hourly Averaged Non-Seabreeze Days\n2m Air Temperature (\N{DEGREE SIGN}C)\n{} to {}'.format(sb_t0str, sb_t1str)

    else:
        ttl = '2m Air Temperature (\N{DEGREE SIGN}C)\n{}'.format(sb_t0str)

    levels = np.arange(16, 37, 1)
    #ticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    #fig, ax = plt.subplots(figsize=(9, 8))
    fig, ax = plt.subplots(figsize=(9, 5))

    # initialize keyword arguments for plotting
    kwargs = dict()
    # kwargs['levels'] = levels  # for contourf only
    #kwargs['cbar_ticks'] = ticks
    cmap = cmo.cm.thermal  # for pcolormesh only
    kwargs['cmap'] = cmap  # for pcolormesh only
    levels = levels  # for pcolormesh only
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)  # for pcolormesh only
    kwargs['norm_clevs'] = norm  # for pcolormesh only

    if '_perpendicular' in line:
        xlab = 'Distance From Shore (km)'
        xvar = distance_km
        wea1 = 14.73  # lines for WEA
        wea2 = 36.32  # lines for WEA
    else:
        xlab = 'Longitude'
        xvar = lons_interp
        wea1 = -74.45  # for longitude: edges of the WEA
        wea2 = -73.95  # for longitude: edges of the WEA

    kwargs['ttl'] = ttl
    kwargs['title_size'] = 12
    kwargs['clab'] = 'Air Temperature (\N{DEGREE SIGN}C)'
    kwargs['cax_size'] = '3%'
    kwargs['xlab'] = xlab
    kwargs['ylab'] = 'Hour (EDT)'
    kwargs['extend'] = 'neither'
    #kwargs['yticks'] = [5, 10, 15, 20]
    pf.plot_pcolormesh(fig, ax, xvar, hours - 4, t2_final, **kwargs)

    ylims = ax.get_ylim()

    if '_perpendicular' in line:
        # add a line for the coast
        # #ax.vlines(coastline_lon, ylims[0], ylims[1], colors='k', ls='--')
        ax.vlines(0, ylims[0], ylims[1], colors='k', ls='-')

    # add lines for the wind energy area (calculated in hovmoller_line_map.py)
    ax.vlines(wea1, ylims[0], ylims[1], colors='darkgray', ls='--')
    ax.vlines(wea2, ylims[0], ylims[1], colors='darkgray', ls='--')

    # ax.set_ylim(ylims)
    #ax.set_xlim([-200, 200])

    sname = 'airtemp_hovmoller.png'
    if interval_name == 'airtemp_hovmoller_zoomed':
        sname = f'{sname.split(".png")[0]}_{pd.to_datetime(sb_t0str).strftime("%Y%m%d")}.png'
    elif interval_name == 'airtemp_hourly_avg_hovmoller_zoomed_seabreeze':
        sname = 'airtemp_hovmoller_hourlyavg_seabreeze.png'
    elif interval_name == 'airtemp_hourly_avg_hovmoller_zoomed_noseabreeze':
        sname = 'airtemp_hovmoller_hourlyavg_noseabreeze.png'
    plt.savefig(os.path.join(save_dir, sname), dpi=200)
    plt.close()


def main(sDir, sdate, edate, intvl, line):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    if intvl == 'airtemp_hourly_cases_hovmoller_zoomed':
        savedir = os.path.join(sDir, 'hovmoller_seabreeze_cases', '{}_{}'.format(intvl, sdate.strftime('%Y%m%d')))
    else:
        savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    if line == 'wea':
        savedir = f'{savedir}_wea'
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    if intvl == 'airtemp_hourly_avg_hovmoller_zoomed':
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # grab the WRF data for the seabreeze dates
        ds_sb = ds.sel(time=sb_datetimes)
        # ds_sb = ds.sel(time=slice(dt.datetime(2020, 6, 1, 13, 0), dt.datetime(2020, 6, 1, 15, 0)))  # for debugging

        # grab the WRF data for the non-seabreeze dates
        nosb_datetimes = [t for t in ds.time.values if t not in sb_datetimes]
        ds_nosb = ds.sel(time=nosb_datetimes)
        # ds_nosb = ds.sel(time=slice(dt.datetime(2020, 6, 2, 0, 0), dt.datetime(2020, 6, 2, 15, 0)))  # for debugging

        plot_airtemp_hovmoller(ds_sb, savedir, 'airtemp_hourly_avg_hovmoller_zoomed_seabreeze', line, **kwargs)
        plot_airtemp_hovmoller(ds_nosb, savedir, 'airtemp_hourly_avg_hovmoller_zoomed_noseabreeze', line, **kwargs)
    elif intvl == 'airtemp_hovmoller_zoomed':
        # plot airtemp for a series of days
        daterange = pd.date_range(sdate, edate)
        for dr in daterange:
            print(dr)
            ds_dr = ds.sel(time=slice(dr, dr + dt.timedelta(hours=23)))
            kwargs['sb_t0str'] = dr.strftime('%Y-%m-%d')
            kwargs['sb_t1str'] = dr.strftime('%Y-%m-%d')
            # ds_dr = ds.sel(time=slice(dt.datetime(2020, 6, 1, 13, 0), dt.datetime(2020, 6, 1, 15, 0)))  # for debugging
            plot_airtemp_hovmoller(ds_dr, savedir, intvl, line, **kwargs)
    else:
        # ds = ds.sel(time=slice(dt.datetime(2020, 6, 8, 13, 0), dt.datetime(2020, 6, 8, 15, 0)))  # for debugging
        plot_airtemp_hovmoller(ds, savedir, intvl, line, **kwargs)


if __name__ == '__main__':
    # save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2020, 6, 8, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  #dt.datetime(2020, 6, 8, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'airtemp_hovmoller_zoomed'  # 'airtemp_hourly_avg_hovmoller_zoomed' 'airtemp_hovmoller_zoomed' - use this for daily plots
    line = 'short_perpendicular'   # 'short_perpendicular'  'wea'
    main(save_directory, start_date, end_date, interval, line)
