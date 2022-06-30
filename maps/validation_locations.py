#!/usr/bin/env python

"""
Author: Lori Garzio on 6/28/2022
Last modified: 6/30/2022
Plot surface map of the WRF validation locations
"""

import matplotlib.pyplot as plt
import functions.configs as configs
import functions.hurricanes_plotting as hp
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def add_validation_points(axis, text, values, zoom=None):
    zoom = zoom or False

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

    if zoom:
        lon_offset = values['text_offset_zoom'][0]
        lat_offset = values['text_offset_zoom'][1]
    else:
        lon_offset = values['text_offset'][0]
        lat_offset = values['text_offset'][1]

    text_lon = values['lon'] + lon_offset
    text_lat = values['lat'] + lat_offset
    axis.text(text_lon, text_lat, text, fontsize=10, transform=configs.projection_merc['data'], zorder=20)


def main(save_file, shape_files, ndbc, extent, zoom):
    # set up the map
    fig, ax = hp.map_create(extent)

    if shape_files:
        lease = '/Users/garzio/Documents/rucool/bpu/wrf/lease_areas/BOEM-Renewable-Energy-Shapefiles_6_1_2022/BOEMWindLeaseOutlines_6_1_2022.shp'
        plan = '/Users/garzio/Documents/rucool/bpu/wrf/lease_areas/BOEM-Renewable-Energy-Shapefiles_6_1_2022/BOEMWindPlanningAreaOutlines_6_1_2022.shp'
        kwargs = dict()
        kwargs['edgecolor'] = 'dimgray'
        hp.map_add_boem_outlines(ax, lease, **kwargs)

        kwargs['edgecolor'] = 'lightgray'
        hp.map_add_boem_outlines(ax, plan, **kwargs)

    kwargs = dict()
    kwargs['zoom'] = zoom
    for key, v in configs.validation_points.items():
        add_validation_points(ax, key, v, **kwargs)

    if ndbc:
        for key, v in configs.ndbc.items():
            add_validation_points(ax, key, v, **kwargs)

    # add a legend, only showing one set of legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), framealpha=.8, loc='lower right')

    plt.savefig(save_file, dpi=200)
    plt.close()


if __name__ == '__main__':
    savefile = '/Users/garzio/Documents/rucool/bpu/wrf/wrf_validation_points_20220629.png'
    shpfiles = True  # True False
    add_ndbc = False  # True False
    extent = [-74.9, -72.3, 38.7, 40.3]  # zoomed in to WEA
    # extent = [-75.5, -72, 38.4, 40.6]
    zoom_wea = True
    main(savefile, shpfiles, add_ndbc, extent, zoom_wea)
