#!/usr/bin/env python

"""
Author: Lori Garzio on 4/20/2021
Last modified: 4/20/2021
Creates wind rose plots from WRF data at 10m and 160m at user-defined grouping intervals at two locations:
1) NYSERDA North LiDAR buoy and 2) NYSERDA South LiDAR buoy.
"""

import numpy as np
import os
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import functions.common as cf
import functions.plotting as pf


def new_axes():
    """
    Create new wind rose axes
    """
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.15, 0.15, 0.75, 0.75]
    ax = WindroseAxes(fig, rect, facecolor='w')
    fig.add_axes(ax)
    return ax


def main(sDir, sdate, edate, intvl):
    wrf = 'http://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best'
    heights = [160, 10]

    savedir = os.path.join(sDir, '{}_{}-{}'.format(intvl, sdate.strftime('%Y%m%d'), edate.strftime('%Y%m%d')))
    os.makedirs(savedir, exist_ok=True)

    # locations of NYSERDA LIDAR buoys
    nyserda_buoys = cf.nyserda_buoys()

    # break up date range into the plotting interval specified
    start, end = cf.daterange_interval(intvl, sdate, edate)

    ds = xr.open_dataset(wrf)
    for sd, ed in zip(start, end):
        # sd = dt.datetime(2019, 9, 1, 0, 0)  # for debugging
        # ed = dt.datetime(2019, 9, 2, 0, 0)  # for debugging
        dst = ds.sel(time=slice(sd, ed + dt.timedelta(hours=23)))
        lat = dst['XLAT']
        lon = dst['XLONG']
        for height in heights:
            if height == 10:
                u = dst['U10']
                v = dst['V10']
            else:
                u = dst.sel(height=height)['U']
                v = dst.sel(height=height)['V']

            for nb, binfo, in nyserda_buoys.items():
                # Find the closest model point
                # calculate the sum of the absolute value distance between the model location and buoy location
                a = abs(lat - binfo['coords']['lat']) + abs(lon - binfo['coords']['lon'])

                # find the indices of the minimum value in the array calculated above
                i, j = np.unravel_index(a.argmin(), a.shape)

                # grab the data just at that location
                usub = u[:, i, j]
                vsub = v[:, i, j]

                ax = new_axes()
                plt_title = 'RU-WRF 4.1 at {}\n{}: {}m'.format(binfo['name'], sd.strftime('%b %Y'), str(height))
                sname = 'WRF_windrose_{}_{}m_{}.png'.format(binfo['code'], height, sd.strftime('%Y%m%d'))
                sfile = os.path.join(savedir, sname)

                ws = cf.wind_uv_to_spd(usub.values, vsub.values)
                wdir = cf.wind_uv_to_dir(usub.values, vsub.values)

                pf.plot_windrose(ax, ws, wdir, plt_title)

                plt.savefig(sfile, dpi=150)
                plt.close()


if __name__ == '__main__':
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windrose'
    save_dir = '/www/home/lgarzio/public_html/bpu/windrose'  # on server
    start_date = dt.datetime(2019, 9, 1, 0, 0)
    end_date = dt.datetime(2020, 9, 1, 0, 0)
    interval = 'monthly'
    main(save_dir, start_date, end_date, interval)
