#!/usr/bin/env python

"""
Author: Lori Garzio on 10/26/2021
Last modified: 12/14/2021
Plot Hovmoller diagram of hourly-averaged wind speed divergence at specified cross-section
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
import metpy.calc as mc
from wrf import interpline, CoordPair, WrfProj
from geographiclib.geodesic import Geodesic
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plot_divergence_hovmoller(ds_sub, save_dir, interval_name, line, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250, 200, 160, 10]

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

    for height in heights:
        print('plotting {}m'.format(height))
        if height == 10:
            u = ds_sub['U10']
            v = ds_sub['V10']
        else:
            u = ds_sub.sel(height=height)['U']
            v = ds_sub.sel(height=height)['V']

        hours = np.arange(13, 23)

        divergence = np.empty(shape=(len(hours), div_shape))

        divergence[:] = np.nan

        for hour_idx, hour in enumerate(hours):
            u_hour = u[u.time.dt.hour == hour]
            v_hour = v[v.time.dt.hour == hour]

            # calculate hourly average
            uhm = u_hour.mean('time')
            vhm = v_hour.mean('time')

            # calculate divergence
            div = mc.divergence(uhm, vhm) * 10**4  # surface divergence, *10^-4 1/s

            # get values along specified line
            wrf_projection = WrfProj(map_proj=1, dx=3000, dy=3000, truelat1=38.8, truelat2=38.8,
                                     moad_cen_lat=38.75293, stand_lon=-74.5)
            ll_point = CoordPair(lat=34.321144, lon=-79.763733)
            div_line = interpline(div, start_point=point_start, end_point=point_end, projection=wrf_projection,
                                  ll_point=ll_point, latlon=True)

            # get the coordinates for the line that is returned
            if hour_idx == 0:
                # get the bathymetry along the interpolated line

                lats_interp = np.array([])
                lons_interp = np.array([])
                land_mask = np.array([])
                for i, value in enumerate(div_line.xy_loc.values):
                    lats_interp = np.append(lats_interp, value.lat)
                    lons_interp = np.append(lons_interp, value.lon)

                    # find the land mask at the closest grid point
                    # calculate the sum of the absolute value distance between the model location and buoy location
                    a = abs(lat - value.lat) + abs(lon - value.lon)

                    # find the indices of the minimum value in the array calculated above
                    i, j = np.unravel_index(a.argmin(), a.shape)
                    land_mask = np.append(land_mask, lm[i, j].values)

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

            divergence[hour_idx] = div_line

        if interval_name == 'divergence_hourly_avg_hovmoller':
            ttl = 'Hourly Averaged Seabreeze Days\nDivergence Along Cross-Section: {}m\n{} to {}'.format(height, sb_t0str, sb_t1str)
            levels = [-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
                      0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
            ticks = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
        else:
            ttl = 'Divergence Along Cross_Section: {}m\n{}'.format(height, sb_t0str)
            levels = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            ticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

        #fig, ax = plt.subplots(figsize=(9, 8))
        fig, ax = plt.subplots(figsize=(9, 5))

        # initialize keyword arguments for plotting
        kwargs = dict()
        # kwargs['levels'] = levels  # for contourf only
        kwargs['cbar_ticks'] = ticks
        cmap = plt.get_cmap('RdBu_r')  # for pcolormesh only
        kwargs['cmap'] = cmap  # for pcolormesh only
        levels = levels  # for pcolormesh only
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)  # for pcolormesh only
        kwargs['norm_clevs'] = norm  # for pcolormesh only

        kwargs['ttl'] = ttl
        kwargs['title_size'] = 12
        kwargs['clab'] = 'Divergence x $10^{-4}$ (1/s)'
        kwargs['cax_size'] = '2%'
        kwargs['xlab'] = 'Distance From Shore (km)'
        #kwargs['xlab'] = 'Longitude'
        kwargs['ylab'] = 'Hour (EDT)'
        #kwargs['yticks'] = [5, 10, 15, 20]
        pf.plot_pcolormesh(fig, ax, distance_km, hours - 4, divergence, **kwargs)

        ylims = ax.get_ylim()
        # # add a line for the coast
        # #ax.vlines(coastline_lon, ylims[0], ylims[1], colors='k', ls='--')
        ax.vlines(0, ylims[0], ylims[1], colors='k', ls='-')

        # add lines for the wind energy area (calculated in hovmoller_line_map.py)
        wea1 = 14.73  # for distance from shore
        wea2 = 36.32  # for distance from shore
        #wea1 = -74.45  # for longitude: edges of the WEA
        #wea2 = -73.95  # for longitude: edges of the WEA
        ax.vlines(wea1, ylims[0], ylims[1], colors='darkgray', ls='--')
        ax.vlines(wea2, ylims[0], ylims[1], colors='darkgray', ls='--')

        # ax.set_ylim(ylims)
        #ax.set_xlim([-200, 200])

        sname = 'divergence_hovmoller_{}.png'.format(height)
        plt.savefig(os.path.join(save_dir, sname), dpi=200)
        plt.close()


def main(sDir, sdate, edate, intvl, line):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    if intvl == 'divergence_hourly_cases_hovmoller_zoomed':
        savedir = os.path.join(sDir, 'hovmoller_seabreeze_cases', '{}_{}'.format(intvl, sdate.strftime('%Y%m%d')))
    else:
        savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    ds = xr.open_dataset(wrf)
    ds = ds.sel(time=slice(sdate, edate))
    dst0 = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')
    dst1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m-%d')

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    if intvl == 'divergence_hourly_avg_hovmoller':
        df = pd.read_csv(os.path.join(sDir, 'radar_seabreezes_2020.csv'))
        df = df[df['Seabreeze'] == 'y']
        sb_dates = np.array(pd.to_datetime(df['Date']))
        sb_datetimes = [pd.date_range(pd.to_datetime(x), pd.to_datetime(x) + dt.timedelta(hours=23), freq='H') for x in sb_dates]
        sb_datetimes = pd.to_datetime(sorted([inner for outer in sb_datetimes for inner in outer]))

        # grab the WRF data for the seabreeze dates
        ds = ds.sel(time=sb_datetimes)
        # ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 5, 0)))  # for debugging

        plot_divergence_hovmoller(ds, savedir, intvl, line, **kwargs)
    else:
        # ds = ds.sel(time=slice(dt.datetime(2020, 6, 8, 13, 0), dt.datetime(2020, 6, 8, 15, 0)))  # for debugging
        plot_divergence_hovmoller(ds, savedir, intvl, line, **kwargs)


if __name__ == '__main__':
    #save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 8, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 6, 8, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'divergence_hourly_cases_hovmoller_zoomed'    # divergence_hourly_avg_hovmoller  divergence_hourly_cases_hovmoller - use this for seabreeze cases
    line = 'short_perpendicular'
    main(save_directory, start_date, end_date, interval, line)
