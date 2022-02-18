#!/usr/bin/env python

"""
Author: Lori Garzio on 2/16/2022
Last modified: 2/17/2022
Plot U*
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


def main(fdir, savedir):
    files = sorted(glob.glob(fdir + '*.nc'))
    plt_region = cf.plot_regions('1km')
    extent = plt_region['windturb']['extent']
    xticks = plt_region['windturb']['xticks']
    yticks = plt_region['windturb']['yticks']
    plt_vars = ['ustar', 'ustar_squared']

    for fname in files:
        for pv in plt_vars:

            splitter = fname.split('/')[-1].split('_')

            save_name = '{}_{}_{}_{}.png'.format(pv, fname.split('/')[-3], splitter[2], splitter[-1].split('.nc')[0])

            sdir = os.path.join(savedir, pv)
            save_file = os.path.join(sdir, save_name)
            os.makedirs(sdir, exist_ok=True)

            ds = xr.open_dataset(fname)

            ust = np.squeeze(ds['UST'])
            levels = list(np.arange(0, 1.05, .05))
            zero_value = 1.e-04
            color_label = 'UST (m/s)'

            if pv == 'ustar_squared':
                ust = ust * ust
                color_label = ' '.join((r'$\rmUST^{2}$', r'$\rm (m^{2}/s^{2}$)'))
                levels = list(np.arange(0, .525, .025))
                zero_value = 9.999999e-09

            mask = np.logical_and(ust == zero_value, ust == zero_value)
            ust.values[mask] = np.nan

            lon = ust.XLONG.values
            lat = ust.XLAT.values

            # set up the map
            lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))
            pf.add_map_features(ax, extent, xticks=xticks, yticks=yticks)

            la_polygon, pa_polygon = cf.extract_lease_area_outlines()
            kwargs = dict()
            kwargs['lw'] = 1.2
            pf.add_lease_area_polygon(ax, la_polygon, 'magenta', **kwargs)  # lease areas  '#969696'  '#737373'

            # set color map
            cmap = plt.get_cmap('YlGnBu')

            kwargs = dict()
            kwargs['ttl'] = '{} {}'.format(color_label, pd.to_datetime(ds.Time.values[0]).strftime('%Y-%m-%d %H:%M'))
            kwargs['cmap'] = cmap
            kwargs['clab'] = color_label
            kwargs['levels'] = levels
            kwargs['extend'] = 'max'
            pf.plot_contourf(fig, ax, lon, lat, ust, **kwargs)

            plt.savefig(save_file, dpi=200)
            plt.close()


if __name__ == '__main__':
    # file_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmrun/20220116/'
    file_dir = '/home/lgarzio/rucool/bpu/wrf/windturbs/wrfout_windturbs/1kmctrl/20220116/'  # server
    # save_dir = '/Users/garzio/Documents/rucool/bpu/wrf/windturbs/plots/'
    save_dir = '/www/home/lgarzio/public_html/bpu/windturbs/'  # server
    main(file_dir, save_dir)
