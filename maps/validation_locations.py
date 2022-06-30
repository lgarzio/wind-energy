#!/usr/bin/env python

"""
Author: Lori Garzio on 6/28/2022
Last modified: 6/28/2022
Plot surface map of the WRF validation locations
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
import functions.configs as configs
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def add_validation_points(axis, text, values):
    if values['active'] == 'Current':
        marker = 'o'
        color = 'magenta'
        size = 40
    else:
        marker = 'X'
        color = 'cyan'
        size = 50
    axis.scatter(values['lon'], values['lat'], s=size, marker=marker, c=color, edgecolor='k', label=values['active'],
                 transform=configs.projection_merc['data'], zorder=20)
    text_lon = values['lon'] + values['text_offset'][0]
    text_lat = values['lat'] + values['text_offset'][1]
    axis.text(text_lon, text_lat, text, fontsize=10, transform=configs.projection_merc['data'], zorder=20)


def extract_shapefiles(shpfile):
    areas = gpd.read_file(shpfile)
    areas = areas.to_crs(crs={'init': 'epsg:4326'})

    return areas


def main(save_file, shape_files, ndbc):
    #extent = [-75.5, -72.1, 38.7, 40.3]  # zoomed in to WEA
    extent = [-75.5, -72, 38.4, 40.6]

    # set up the map
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=configs.projection_merc['map']))

    kwargs = dict()
    kwargs['landcolor'] = 'tan'
    #kwargs['yticks'] = [39, 39.5, 40]
    kwargs['yticks'] = [38.5, 39, 39.5, 40, 40.5]
    #kwargs['zoom_shore'] = 'full'
    kwargs['add_ocean_color'] = True
    pf.add_map_features(ax, extent, **kwargs)

    if shape_files:
        shape_file_location = '/Users/garzio/Documents/rucool/bpu/wrf/lease_areas/BOEM-Renewable-Energy-Shapefiles_6_1_2022'
        lease = extract_shapefiles(os.path.join(shape_file_location, 'BOEMWindLeaseOutlines_6_1_2022.shp'))
        plan = extract_shapefiles(os.path.join(shape_file_location, 'BOEMWindPlanningAreaOutlines_6_1_2022.shp'))
        lease.plot(ax=ax, color='none', edgecolor='dimgray', transform=configs.projection_merc['data'])
        plan.plot(ax=ax, color='none', edgecolor='lightgray', transform=configs.projection_merc['data'])

    for key, v in configs.validation_points.items():
        add_validation_points(ax, key, v)

    if ndbc:
        for key, v in configs.ndbc.items():
            add_validation_points(ax, key, v)

    # add a legend, only showing one set of legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys(), framealpha=.8, loc='lower right', markerscale=6, ncol=2)
    plt.legend(by_label.values(), by_label.keys(), framealpha=.8, loc='lower right')

    plt.savefig(save_file, dpi=200)
    plt.close()


if __name__ == '__main__':
    savefile = '/Users/garzio/Documents/rucool/bpu/wrf/wrf_validation_points_20220629_ndbc-noshapefiles.png'
    shpfiles = False  # True False
    add_ndbc = True  # True False
    main(savefile, shpfiles, add_ndbc)
