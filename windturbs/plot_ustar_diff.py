#!/usr/bin/env python

"""
Author: Lori Garzio on 2/17/2022
Last modified: 2/22/2022
Plot difference in U*, wind farm minus control WRF output
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(fdir, fdir_ctrl, savedir, plot_turbs):
    files = sorted(glob.glob(fdir + '*.nc'))
    plt_region = cf.plot_regions('1km')
    extent = plt_region['windturb']['extent']
    xticks = plt_region['windturb']['xticks']
    yticks = plt_region['windturb']['yticks']
    plt_vars = ['ustar_diff', 'ustar_squared_diff']

    for fname in files:
        f = fname.split('/')[-1]

        # find the corresponding control file
        fname_ctrl = os.path.join(fdir_ctrl, f)

        splitter = f.split('_')

        for pv in plt_vars:
            save_name = '{}_{}_{}.png'.format(pv, splitter[2], splitter[-1].split('.nc')[0])

            sdir = os.path.join(savedir, pv)
            save_file = os.path.join(sdir, save_name)
            os.makedirs(sdir, exist_ok=True)

            ds = xr.open_dataset(fname)
            ds_ctrl = xr.open_dataset(fname_ctrl)

            ust = np.squeeze(ds['UST'])
            ust_ctrl = np.squeeze(ds_ctrl['UST'])
            levels = [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01,
                      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            cbar_ticks = [-0.1, -0.08, -0.06, -0.04, -0.02,  0, 0.02, 0.04, 0.06, 0.08, 0.1]
            color_label = 'UST Difference (m/s)'

            if pv == 'ustar_squared_diff':
                ust = ust * ust
                ust_ctrl = ust_ctrl * ust_ctrl
                color_label = ' '.join((r'$\rmUST^{2}$ Difference', r'$\rm (m^{2}/s^{2}$)'))

            diff = ust - ust_ctrl

            # mask the land values
            landmask = ds['LANDMASK']  # 1=land, 0=water
            lakemask = ds['LAKEMASK']  # 1=lake, 0=non-lake

            ldmask = np.logical_and(landmask == 1, landmask == 1)
            diff.values[ldmask] = np.nan

            lkmask = np.logical_and(lakemask == 1, lakemask == 1)
            diff.values[lkmask] = np.nan

            # create a masked array
            masked_diff = np.ma.masked_inside(diff, -0.01, 0.01)

            # mask = np.logical_and(diff == 0, diff == 0)
            # diff.values[mask] = np.nan

            lon = ust.XLONG.values
            lat = ust.XLAT.values

            # set up the map
            lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
            pf.add_map_features(ax, extent, xticks=xticks, yticks=yticks, landcolor='tan', zoom_shore=True)

            la_polygon, pa_polygon = cf.extract_lease_area_outlines()
            kwargs = dict()
            kwargs['lw'] = 1.2
            pf.add_lease_area_polygon(ax, la_polygon, '#969696', **kwargs)  # lease areas  '#969696'  '#737373'

            # set color map
            cmap = plt.get_cmap('RdBu_r')
            cmap.set_bad('white')

            kwargs = dict()
            kwargs['ttl'] = '{} {}'.format(color_label, pd.to_datetime(ds.Time.values[0]).strftime('%Y-%m-%d %H:%M'))
            kwargs['cmap'] = cmap
            kwargs['clab'] = color_label
            kwargs['levels'] = levels
            kwargs['extend'] = 'both'
            kwargs['cbar_ticks'] = cbar_ticks
            pf.plot_contourf(fig, ax, lon, lat, masked_diff, **kwargs)

            if plot_turbs:
                df = pd.read_csv(plot_turbs)
                ax.scatter(df.lon, df.lat, s=.5, color='k', transform=ccrs.PlateCarree())

            plt.savefig(save_file, dpi=200)
            plt.close()


if __name__ == '__main__':
    file_dir = '/home/lgarzio/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20210901/'  # server
    file_dir_ctrl = '/home/lgarzio/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20210901/'  # server
    save_dir = '/www/home/lgarzio/public_html/bpu/windturbs/20210901/'  # server
    plot_turbines = '/www/home/lgarzio/public_html/bpu/windturbs/turbine_locations_final.csv'  # server
    # file_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20210901/'
    # file_dir_ctrl = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20210901/'
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/20210901/'
    # plot_turbines = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/turbine_locations_final.csv'
    main(file_dir, file_dir_ctrl, save_dir, plot_turbines)
