#!/usr/bin/env python

"""
Author: Lori Garzio on 10/26/2021
Last modified: 11/11/2021
Plot map of WRF interpolated line
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


def plot_divergence_hovmoller(ds_sub, save_dir, interval_name, t0=None, sb_t0str=None, sb_t1str=None):
    t0 = t0 or None
    sb_t0str = sb_t0str or None
    sb_t1str = sb_t1str or None
    heights = [250]

    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    extent = [-76.5, -72.3, 37.6, 41]
    bathy = xr.open_dataset(bathymetry)
    bathy = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                      lat=slice(extent[2] - .1, extent[3] + .1))

    lon = ds_sub.XLONG.values
    lat = ds_sub.XLAT.values

    lm = ds_sub.LANDMASK.mean('time')

    # grab data along the line perpendicular to the coast in southern NJ
    point_start = CoordPair(lat=39.8, lon=-74.95)
    point_end = CoordPair(lat=38.85, lon=-73.8)
    point_start2 = CoordPair(lat=40.7, lon=-76)
    point_end2 = CoordPair(lat=38, lon=-72.8)

    for height in heights:
        if height == 10:
            u = ds_sub['U10']
            v = ds_sub['V10']
        else:
            u = ds_sub.sel(height=height)['U']
            v = ds_sub.sel(height=height)['V']

        hours = [1]

        divergence = np.empty(shape=(len(hours), 49))
        divergence[:] = np.nan

        for hour in hours:
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
            div_line2 = interpline(div, start_point=point_start2, end_point=point_end2, projection=wrf_projection,
                                  ll_point=ll_point, latlon=True)

            # get the coordinates for the line that is returned
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
            coastline_idx = np.where(land_mask[:-1] != land_mask[1:])[0]
            coastline_lon = np.mean(lons_interp[coastline_idx[0]:coastline_idx[0] + 2])
            coastline_lat = np.mean(lats_interp[coastline_idx[0]:coastline_idx[0] + 2])

            # calculate the distance from each coordinate to the coastline
            # negative values are land-side, positive values are ocean-side
            distance_km = np.array([])
            geod = Geodesic.WGS84
            for i, lati in enumerate(lats_interp):
                g = geod.Inverse(coastline_lat, coastline_lon, lati, lons_interp[i])
                dist_km = g['s12'] * .001
                if land_mask[i] == 1:
                    dist_km = -dist_km
                distance_km = np.append(distance_km, dist_km)

            lats_interp2 = np.array([])
            lons_interp2 = np.array([])
            land_mask2 = np.array([])
            for i, value in enumerate(div_line2.xy_loc.values):
                lats_interp2 = np.append(lats_interp2, value.lat)
                lons_interp2 = np.append(lons_interp2, value.lon)

                # find the land mask at the closest grid point
                # calculate the sum of the absolute value distance between the model location and buoy location
                a = abs(lat - value.lat) + abs(lon - value.lon)

                # find the indices of the minimum value in the array calculated above
                i, j = np.unravel_index(a.argmin(), a.shape)
                land_mask2 = np.append(land_mask2, lm[i, j].values)

            # find the coastline longitude (where landmask values change from 1 to 0)
            coastline_idx2 = [np.nanmax(np.where(land_mask2[:-1] != land_mask2[1:])[0])]
            coastline_lon2 = np.mean(lons_interp2[coastline_idx2[0]:coastline_idx2[0] + 2])
            coastline_lat2 = np.mean(lats_interp2[coastline_idx2[0]:coastline_idx2[0] + 2])

            # calculate the distance from each coordinate to the coastline
            # negative values are land-side, positive values are ocean-side
            distance_km2 = np.array([])
            geod = Geodesic.WGS84
            for i, lati in enumerate(lats_interp2):
                g = geod.Inverse(coastline_lat2, coastline_lon2, lati, lons_interp2[i])
                dist_km = g['s12'] * .001
                if land_mask2[i] == 1:
                    dist_km = -dist_km
                distance_km2 = np.append(distance_km2, dist_km)
            print(distance_km2)

            # set up map
            lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))

            # define bathymetry levels and data
            # levels = np.arange(-5000, 5100, 50)
            bath_lat = bathy.variables['lat'][:]
            bath_lon = bathy.variables['lon'][:]
            bath_elev = bathy.variables['elevation'][:]
            # plt.contourf(bath_lon, bath_lat, bath_elev,  levels, cmap=cmo.cm.topo, transform=ccrs.PlateCarree())

            levels = [-3000, -2000, -1000, -100]
            CS = plt.contour(bath_lon, bath_lat, bath_elev, levels, linewidths=.75, alpha=.5, colors='k',
                             transform=ccrs.PlateCarree())
            ax.clabel(CS, levels, inline=True, fontsize=8, fmt='%d')

            pf.add_map_features(ax, extent)
            la_polygon, pa_polygon = cf.extract_lease_area_outlines()
            pf.add_lease_area_polygon(ax, la_polygon, 'tab:green')  # lease areas
            ax.plot(lons_interp2, lats_interp2, transform=ccrs.PlateCarree())
            #ax.plot(lons_interp, lats_interp, lw=5, transform=ccrs.PlateCarree())
            #ax.plot(lons_interp2, lats_interp2, color='r', transform=ccrs.PlateCarree())

            # # add points to figure out distance from shore for WEA
            # pt1 = [-74.28, 39.28]
            # pt2 = [-74.12, 39.13]
            # ax.plot(pt1[0], pt1[1], 'ro', transform=ccrs.PlateCarree())
            # ax.plot(pt2[0], pt2[1], 'ro', transform=ccrs.PlateCarree())
            # g = geod.Inverse(coastline_lat2, coastline_lon2, pt1[1], pt1[0])
            # dist_km1 = g['s12'] * .001
            # g = geod.Inverse(coastline_lat2, coastline_lon2, pt2[1], pt2[0])
            # dist_km2 = g['s12'] * .001

            sname = f'hovmoller_map.png'
            plt.savefig(os.path.join(save_dir, sname), dpi=200)
            plt.close()


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'

    savedir = os.path.join(sDir, '{}_{}-{}-contourf'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
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

    # grab the WRF data for the seabreeze dates
    ds = ds.sel(time=sb_datetimes)
    ds = ds.sel(time=slice(dt.datetime(2020, 6, 1, 0, 0), dt.datetime(2020, 6, 1, 5, 0)))  # for debugging

    # define arguments for plotting function
    kwargs = dict()
    kwargs['sb_t0str'] = dst0
    kwargs['sb_t1str'] = dst1

    plot_divergence_hovmoller(ds, savedir, intvl, **kwargs)


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    #save_directory = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2020, 6, 1, 0, 0)  # dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 7, 31, 23, 0)  # dt.datetime(2020, 9, 1, 0, 0)
    interval = 'divergence_hourly_avg_hovmoller'
    main(save_directory, start_date, end_date, interval)
