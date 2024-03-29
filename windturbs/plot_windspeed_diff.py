#!/usr/bin/env python

"""
Author: Lori Garzio on 2/17/2022
Last modified: 12/1/2022
Plot wind speed difference at hub height (160m), wind farm minus control WRF output. Plot optional vectors
from the control run.
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


def main(fdir, fdir_ctrl, savedir, plot_vec):
    files = sorted(glob.glob(os.path.join(fdir, '*.nc')))
    if files[0].endswith('_M00.nc'):
        files = sorted(glob.glob(os.path.join(fdir, '*_M00.nc')))
    plt_region = cf.plot_regions('1km')
    extent = plt_region['windturb']['extent']
    xticks = plt_region['windturb']['xticks']
    yticks = plt_region['windturb']['yticks']
    color_label = 'Wind Speed Difference (m/s)'
    heights = [160, 10]

    # get the date from the first file
    ds0 = xr.open_dataset(files[0])
    dirtm = pd.to_datetime(ds0.Time.values[0])

    for fname in files:
        f = fname.split('/')[-1]

        # find the corresponding control file
        fname_ctrl = os.path.join(fdir_ctrl, f)

        run_type = fname.split('/')[-4]
        if 'ctrl' in run_type:
            plot_turbs = False
        else:
            plot_turbs = '/www/web/rucool/windenergy/ru-wrf/windturbs/turbine_locations_final.csv'  # server
            # plot_turbs = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/turbine_locations_final.csv'

        for ht in heights:

            ds = xr.open_dataset(fname)
            ds_ctrl = xr.open_dataset(fname_ctrl)

            tm = pd.to_datetime(ds.Time.values[0])

            save_name = 'windspeed_diff_{}m_{}_H{:03d}.png'.format(ht, tm.strftime('%Y%m%d'), tm.hour)

            if plot_vec:
                sdir = os.path.join(savedir, dirtm.strftime('%Y%m%d'), f'windspeed_{ht}m_diff_vectors')
            else:
                sdir = os.path.join(savedir, dirtm.strftime('%Y%m%d'), f'windspeed_{ht}m_diff')
            save_file = os.path.join(sdir, save_name)
            os.makedirs(sdir, exist_ok=True)

            if ht == 10:
                u = np.squeeze(ds['U10'])
                v = np.squeeze(ds['V10'])
                uctrl = np.squeeze(ds_ctrl['U10'])
                vctrl = np.squeeze(ds_ctrl['V10'])
            else:
                u = np.squeeze(ds.sel(height=ht)['U'])
                v = np.squeeze(ds.sel(height=ht)['V'])
                uctrl = np.squeeze(ds_ctrl.sel(height=ht)['U'])
                vctrl = np.squeeze(ds_ctrl.sel(height=ht)['V'])

            # standardize the vectors so they only represent direction
            u_standardize = u / cf.wind_uv_to_spd(u, v)
            v_standardize = v / cf.wind_uv_to_spd(u, v)

            # calculate wind speed from u and v
            speed = cf.wind_uv_to_spd(u, v)
            lon = speed.XLONG.values
            lat = speed.XLAT.values

            speed_ctrl = cf.wind_uv_to_spd(uctrl, vctrl)

            diff = speed - speed_ctrl

            # create a masked array
            masked_diff = np.ma.masked_inside(diff, -0.1, 0.1)
            # mask = np.logical_and(diff == 0, diff == 0)
            # diff.values[mask] = np.nan

            # set up the map
            lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
            pf.add_map_features(ax, extent, xticks=xticks, yticks=yticks, zoom_shore=True)

            la_polygon, pa_polygon = cf.extract_lease_area_outlines()
            kwargs = dict()
            kwargs['lw'] = 1.2
            pf.add_lease_area_polygon(ax, la_polygon, '#969696', **kwargs)  # lease areas  '#969696'  '#737373'

            # set color map
            cmap = plt.get_cmap('RdBu_r')
            cmap.set_bad('white')
            levels = list(np.arange(-2, 2.1, .1))
            #levels = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

            kwargs = dict()
            kwargs['ttl'] = 'Wind Speed Difference ({}m) {}'.format(ht, pd.to_datetime(ds.Time.values[0]).strftime('%Y-%m-%d %H:%M'))
            kwargs['cmap'] = cmap
            kwargs['clab'] = color_label
            kwargs['levels'] = levels
            kwargs['extend'] = 'both'
            #kwargs['cbar_ticks'] = [-3, -2, -1, 0, 1, 2, 3]
            pf.plot_contourf(fig, ax, lon, lat, masked_diff, **kwargs)

            if plot_vec:
                qs = 5
                ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_standardize.values[::qs, ::qs],
                          v_standardize.values[::qs, ::qs], scale=30, width=.002, headlength=4,
                          transform=ccrs.PlateCarree())

            if plot_turbs:
                df = pd.read_csv(plot_turbs)
                ax.scatter(df.lon, df.lat, s=.5, color='k', transform=ccrs.PlateCarree())

            plt.savefig(save_file, dpi=200)
            plt.close()


if __name__ == '__main__':
    file_dir = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout_windturbs/constdt/1km_wf2km/processed/20220810'  # server
    file_dir_ctrl = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout_windturbs/constdt/1km_ctrl/processed/20220810'  # server
    save_dir = '/www/web/rucool/windenergy/ru-wrf/windturbs'  # server
    # file_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20210901/'
    # file_dir_ctrl = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20210901/'
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/20210901/'
    plot_vectors = False
    main(file_dir, file_dir_ctrl, save_dir, plot_vectors)
