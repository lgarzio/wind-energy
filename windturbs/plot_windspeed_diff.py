#!/usr/bin/env python

"""
Author: Lori Garzio on 2/17/2022
Last modified: 2/17/2022
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
    files = sorted(glob.glob(fdir + '*.nc'))
    plt_region = cf.plot_regions('1km')
    extent = plt_region['windturb']['extent']
    xticks = plt_region['windturb']['xticks']
    yticks = plt_region['windturb']['yticks']
    color_label = 'Wind Speed Difference (m/s)'

    for fname in files:
        f = fname.split('/')[-1]

        # find the corresponding control file
        fname_ctrl = os.path.join(fdir_ctrl, f)

        splitter = f.split('_')

        save_name = 'windspeed_diff_{}_{}_{}.png'.format(fname.split('/')[-3], splitter[2], splitter[-1].split('.nc')[0])

        if plot_vec:
            sdir = os.path.join(savedir, 'windspeed_160m_diff_vectors')
        else:
            sdir = os.path.join(savedir, 'windspeed_160m_diff')
        save_file = os.path.join(sdir, save_name)
        os.makedirs(sdir, exist_ok=True)

        ds = xr.open_dataset(fname)
        ds_ctrl = xr.open_dataset(fname_ctrl)

        u = np.squeeze(ds.sel(height=160)['U'])
        v = np.squeeze(ds.sel(height=160)['V'])

        # standardize the vectors so they only represent direction
        u_standardize = u / cf.wind_uv_to_spd(u, v)
        v_standardize = v / cf.wind_uv_to_spd(u, v)

        uctrl = np.squeeze(ds_ctrl.sel(height=160)['U'])
        vctrl = np.squeeze(ds_ctrl.sel(height=160)['V'])

        # calculate wind speed from u and v
        speed = cf.wind_uv_to_spd(u, v)
        lon = speed.XLONG.values
        lat = speed.XLAT.values

        speed_ctrl = cf.wind_uv_to_spd(uctrl, vctrl)

        diff = speed - speed_ctrl

        mask = np.logical_and(diff == 0, diff == 0)
        diff.values[mask] = np.nan

        # set up the map
        lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
        pf.add_map_features(ax, extent, xticks=xticks, yticks=yticks)

        la_polygon, pa_polygon = cf.extract_lease_area_outlines()
        kwargs = dict()
        kwargs['lw'] = 1.2
        pf.add_lease_area_polygon(ax, la_polygon, '#969696', **kwargs)  # lease areas  '#969696'  '#737373'

        # set color map
        cmap = plt.get_cmap('RdBu')
        levels = list(np.arange(-3, 3.5, .5))

        kwargs = dict()
        kwargs['ttl'] = '{} {}'.format(color_label, pd.to_datetime(ds.Time.values[0]).strftime('%Y-%m-%d %H:%M'))
        kwargs['cmap'] = cmap
        kwargs['clab'] = color_label
        kwargs['levels'] = levels
        kwargs['extend'] = 'both'
        pf.plot_contourf(fig, ax, lon, lat, diff, **kwargs)

        if plot_vec:
            qs = 5
            ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_standardize.values[::qs, ::qs],
                      v_standardize.values[::qs, ::qs], scale=30, width=.002, headlength=4,
                      transform=ccrs.PlateCarree())

        plt.savefig(save_file, dpi=200)
        plt.close()


if __name__ == '__main__':
    file_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20220116/'
    file_dir_ctrl = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20220116/'
    save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/'
    plot_vectors = True
    main(file_dir, file_dir_ctrl, save_dir, plot_vectors)
