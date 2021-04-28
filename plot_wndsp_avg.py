#!/usr/bin/env python

"""
Author: Lori Garzio on 4/12/2021
Last modified: 4/28/2021
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

    plt_regions = cf.plot_regions()
    color_label = 'Average Wind Speed (m/s)'

    la_polygon = cf.extract_lease_areas()

    savedir = os.path.join(sDir, '{}_{}-{}-withquiver'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
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

            ttl = 'Average Wind Speed {}m: {}'.format(height, sd.strftime('%b %Y'))

            ws = cf.wind_uv_to_spd(u, v)
            mws = ws.mean('time')

            u_mean = u.mean('time')
            v_mean = v.mean('time')

            # standardize the vectors so they only represent direction
            u_mean_standardize = u_mean / cf.wind_uv_to_spd(u_mean, v_mean)
            v_mean_standardize = v_mean / cf.wind_uv_to_spd(u_mean, v_mean)

            for pr, items in plt_regions.items():
                region_savedir = os.path.join(savedir, pr)
                os.makedirs(region_savedir, exist_ok=True)

                sname = '{}_meanws_{}m_{}_{}'.format(pr, height, intvl, sd.strftime('%Y%m%d'))
                sfile = os.path.join(region_savedir, sname)
                # sfile = os.path.join(savedir, sname)  # for debugging

                # set up the map
                lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
                pf.add_map_features(ax, items['extent'], items['xticks'], items['yticks'])

                # subset grid
                if items['subset']:
                    extent = np.add(items['extent'], [-.5, .5, -.5, .5]).tolist()
                    mws = cf.subset_grid(mws, extent)
                    u_mean_standardize = cf.subset_grid(u_mean_standardize, extent)
                    v_mean_standardize = cf.subset_grid(v_mean_standardize, extent)

                # add lease areas
                if items['lease_area']:
                    pf.add_lease_area_polygon(ax, la_polygon, 'magenta')

                lon = mws.XLONG.values
                lat = mws.XLAT.values

                # plot data
                # pcolormesh: coarser resolution, shows the actual resolution of the model data
                vmin = items['limits']['_{}m'.format(height)]['vmin']
                vmax = items['limits']['_{}m'.format(height)]['vmax']
                pf.plot_pcolormesh(fig, ax, ttl, lon, lat, mws, vmin, vmax, plt.get_cmap('viridis'), color_label)

                # subset the quivers and add as a layer
                if items['quiver_subset']:
                    quiver_scale = items['quiver_scale']
                    qs = items['quiver_subset']['_{}m'.format(height)]
                    ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_mean_standardize.values[::qs, ::qs],
                              v_mean_standardize.values[::qs, ::qs], scale=quiver_scale, transform=ccrs.PlateCarree())
                else:
                    ax.quiver(lon, lat, u_mean_standardize.values, v_mean_standardize.values, scale=quiver_scale,
                              transform=ccrs.PlateCarree())

                plt.savefig(sfile, dpi=200)
                plt.close()


if __name__ == '__main__':
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windspeed_averages'
    save_dir = '/www/home/lgarzio/public_html/bpu/windspeed_averages'  # on server
    start_date = dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 9, 1, 0, 0)
    interval = 'monthly'
    main(save_dir, start_date, end_date, interval)
