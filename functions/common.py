#! /usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 5/6/2021
"""

import numpy as np
import pandas as pd
import datetime as dt
import xml.etree.ElementTree as ET  # for parsing kml files


def daterange_interval(interval, start_date, end_date):
    """
    Break up date range into the plotting interval specified
    """
    if interval == 'monthly':
        daterange = pd.date_range(start_date, end_date, freq='M')
        start = []
        end = []
        for i, dr in enumerate(daterange):
            if i == 0:
                start.append(start_date)
            else:
                start.append((daterange[i - 1] + dt.timedelta(days=1)))
            end.append(dr)
        if dr + dt.timedelta(days=1) != end_date:
            start.append((dr + dt.timedelta(days=1)))
            end.append(end_date)

    return start, end


def extract_lease_area_outlines():
    """
    Extracts outlines of wind energy area polygon coordinates from a .kml file.
    :returns dictionary containing lat/lon coordinates for wind energy lease area polygons, separated by company
    """
    # boem_lease_areas = '/Users/garzio/Documents/rucool/bpu/wrf/lease_areas/boem_lease_area_full.kml'  # on local machine
    # boem_lease_areas = '/Users/garzio/Documents/rucool/bpu/wrf/lease_areas/boem_lease_areas_AS_OW_split.kml'  # on local machine
    boem_lease_areas = '/home/coolgroup/bpu/mapdata/shapefiles/RU-WRF_Plotting_Shapefiles/boem_lease_areas_AS_OW_split.kml'
    nmsp = '{http://www.opengis.net/kml/2.2}'
    doc = ET.parse(boem_lease_areas)
    findouter = './/{0}outerBoundaryIs/{0}LinearRing/{0}coordinates'.format(nmsp)
    findinner = './/{0}innerBoundaryIs/{0}LinearRing/{0}coordinates'.format(nmsp)

    polygon_dict = dict()
    for pm in doc.iterfind('.//{0}Placemark'.format(nmsp)):
        for nm in pm.iterfind('{0}name'.format(nmsp)):  # find the company name
            polygon_dict[nm.text] = dict(outer=[], inner=[])
            polygon_dict[nm.text]['outer'] = find_coords(pm, findouter)
            polygon_dict[nm.text]['inner'] = find_coords(pm, findinner)

    return polygon_dict


def find_coords(elem, findstr):
    """
    Finds coordinates in an .xml file and appends them in pairs to a list
    :param elem: element of an .xml file
    :param findstr: string to find in the element
    :returns list of coordinates
    """
    coordlist = []
    for ls in elem.iterfind(findstr):
        coord_strlst = [x for x in ls.text.split(' ')]
        for coords in coord_strlst:
            splitter = coords.split(',')
            coordlist.append([np.float(splitter[0]), np.float(splitter[1])])

    return coordlist


def plot_regions(plot_version):
    regions = dict(
        full_grid=dict(quiver_subset=dict(_10m=11, _160m=13),
                       quiver_scale=45,
                       extent=[-79.79, -69.2, 34.5, 43],
                       xticks=[-78, -76, -74, -72, -70],
                       yticks=[36, 38, 40, 42],
                       subset=False,
                       lease_area=False),
        mab=dict(quiver_subset=dict(_10m=7, _160m=8),
                 quiver_scale=40,
                 extent=[-77.2, -69.6, 36, 41.8],
                 xticks=[-75, -73, -71],
                 yticks=[37, 39, 41],
                 subset=True,
                 lease_area=True),
        nj=dict(quiver_subset=dict(_10m=4, _160m=5),
                quiver_scale=40,
                extent=[-75.7, -71.5, 38.1, 41.2],
                xticks=[-75, -74, -73, -72],
                yticks=[39, 40, 41],
                subset=True,
                lease_area=True),
        southern_nj=dict(quiver_subset=dict(_10m=3, _160m=3),
                         quiver_scale=40,
                         extent=[-75.6, -73, 38.6, 40.5],
                         xticks=[-75, -74.5, -74, -73.5],
                         yticks=[39, 39.5, 40],
                         subset=True,
                         lease_area=True)
    )

    full_grid = regions['full_grid']
    mab = regions['mab']
    nj = regions['nj']
    snj = regions['southern_nj']
    if 'monthly' in plot_version:
        full_grid.update(
            meanws=dict(limits=dict(_10m=dict(vmin=4, vmax=10, rint=.5), _160m=dict(vmin=6, vmax=14, rint=.5))))
        full_grid.update(
            sdwind=dict(limits=dict(_10m=dict(vmin=6, vmax=12, rint=.5), _160m=dict(vmin=6, vmax=12, rint=.5))))
        full_grid.update(
            sdwind_norm=dict(limits=dict(_10m=dict(vmin=.9, vmax=1.2, rint=.05), _160m=dict(vmin=.9, vmax=1.2, rint=.05))))

        mab.update(meanws=dict(limits=dict(_10m=dict(vmin=4, vmax=10, rint=.5), _160m=dict(vmin=6, vmax=14, rint=.5))))
        mab.update(sdwind=dict(limits=dict(_10m=dict(vmin=6, vmax=12, rint=.5), _160m=dict(vmin=6, vmax=12, rint=.5))))
        mab.update(sdwind_norm=dict(limits=dict(_10m=dict(vmin=.9, vmax=1.2, rint=.05), _160m=dict(vmin=.9, vmax=1.2, rint=.05))))

        nj.update(meanws=dict(limits=dict(_10m=dict(vmin=4, vmax=10, rint=.5), _160m=dict(vmin=6, vmax=14, rint=.5))))
        nj.update(sdwind=dict(limits=dict(_10m=dict(vmin=6, vmax=12, rint=.5), _160m=dict(vmin=6, vmax=12, rint=.5))))
        nj.update(sdwind_norm=dict(limits=dict(_10m=dict(vmin=.9, vmax=1.2, rint=.05), _160m=dict(vmin=.9, vmax=1.2, rint=.05))))

        snj.update(meanws=dict(limits=dict(_10m=dict(vmin=4, vmax=10, rint=.5), _160m=dict(vmin=6, vmax=14, rint=.5))))
        snj.update(sdwind=dict(limits=dict(_10m=dict(vmin=6, vmax=12, rint=.5), _160m=dict(vmin=6, vmax=12, rint=.5))))
        snj.update(sdwind_norm=dict(limits=dict(_10m=dict(vmin=.9, vmax=1.2, rint=.05), _160m=dict(vmin=.9, vmax=1.2, rint=.05))))

    elif 'seabreeze_days' in plot_version:
        full_grid.update(
            meanws=dict(limits=dict(_10m=dict(vmin=2, vmax=10, rint=.5), _160m=dict(vmin=2, vmax=10, rint=.5))))
        full_grid.update(
            sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=8, rint=.5), _160m=dict(vmin=2, vmax=8, rint=.5))))
        full_grid.update(
            sdwind_norm=dict(
                limits=dict(_10m=dict(vmin=.8, vmax=1.2, rint=.05), _160m=dict(vmin=.8, vmax=1.2, rint=.05))))

        mab.update(meanws=dict(limits=dict(_10m=dict(vmin=2, vmax=9, rint=.5), _160m=dict(vmin=2, vmax=9, rint=.5))))
        mab.update(sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=9, rint=.5), _160m=dict(vmin=2, vmax=9, rint=.5))))
        mab.update(sdwind_norm=dict(
            limits=dict(_10m=dict(vmin=.7, vmax=1.2, rint=.05), _160m=dict(vmin=.7, vmax=1.2, rint=.05))))

        nj.update(meanws=dict(limits=dict(_10m=dict(vmin=2, vmax=10, rint=.5), _160m=dict(vmin=2, vmax=10, rint=.5))))
        nj.update(sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=9, rint=.5), _160m=dict(vmin=2, vmax=9, rint=.5))))
        nj.update(sdwind_norm=dict(
            limits=dict(_10m=dict(vmin=.8, vmax=1.2, rint=.05), _160m=dict(vmin=.8, vmax=1.2, rint=.05))))

        snj.update(meanws=dict(limits=dict(_10m=dict(vmin=2, vmax=10, rint=.5), _160m=dict(vmin=2, vmax=10, rint=.5))))
        snj.update(sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=9, rint=.5), _160m=dict(vmin=2, vmax=9, rint=.5))))
        snj.update(sdwind_norm=dict(
            limits=dict(_10m=dict(vmin=.8, vmax=1.2, rint=.05), _160m=dict(vmin=.8, vmax=1.2, rint=.05))))

    elif plot_version in ['seabreeze_morning', 'seabreeze_afternoon']:
        full_grid.update(
            meanws=dict(limits=dict(_10m=dict(vmin=2, vmax=8, rint=.5), _160m=dict(vmin=2, vmax=8, rint=.5))))
        full_grid.update(
            sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=8, rint=.5), _160m=dict(vmin=2, vmax=8, rint=.5))))
        full_grid.update(
            sdwind_norm=dict(
                limits=dict(_10m=dict(vmin=.4, vmax=1.2, rint=.1), _160m=dict(vmin=.4, vmax=1.2, rint=.1))))

        mab.update(meanws=dict(limits=dict(_10m=dict(vmin=2, vmax=8, rint=.5), _160m=dict(vmin=2, vmax=8, rint=.5))))
        mab.update(sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=8, rint=.5), _160m=dict(vmin=2, vmax=8, rint=.5))))
        mab.update(sdwind_norm=dict(
            limits=dict(_10m=dict(vmin=.5, vmax=1.2, rint=.1), _160m=dict(vmin=.5, vmax=1.2, rint=.1))))

        nj.update(meanws=dict(limits=dict(_10m=dict(vmin=1.5, vmax=7, rint=.5), _160m=dict(vmin=1.5, vmax=7, rint=.5))))
        nj.update(sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=8, rint=.5), _160m=dict(vmin=2, vmax=8, rint=.5))))
        nj.update(sdwind_norm=dict(
            limits=dict(_10m=dict(vmin=.6, vmax=1.2, rint=.05), _160m=dict(vmin=.6, vmax=1.2, rint=.05))))

        snj.update(meanws=dict(limits=dict(_10m=dict(vmin=1.5, vmax=7.5, rint=.5), _160m=dict(vmin=1.5, vmax=7.5, rint=.5))))
        snj.update(sdwind=dict(limits=dict(_10m=dict(vmin=2, vmax=7.5, rint=.5), _160m=dict(vmin=2, vmax=7.5, rint=.5))))
        snj.update(sdwind_norm=dict(
            limits=dict(_10m=dict(vmin=.6, vmax=1.2, rint=.05), _160m=dict(vmin=.6, vmax=1.2, rint=.05))))

    return regions


def nyserda_buoys():
    """
    Return dictionary of NYSERDA buoy information
    """
    # locations of NYSERDA LIDAR buoys
    nb = dict(
        nyserda_north=dict(coords=dict(lon=-72.7173, lat=39.9686),
                           name='NYSERDA North',
                           code='NYNE05'),
        nyserda_south=dict(coords=dict(lon=-73.4295, lat=39.5465),
                           name='NYSERDA South',
                           code='NYSE06')
    )

    return nb


def subset_grid(ds, extent):
    """
    Subset the data according to defined latitudes and longitudes, and define the axis limits for the plots
    """
    mlon = ds['XLONG']
    mlat = ds['XLAT']
    lon_ind = np.logical_and(mlon > extent[0], mlon < extent[1])
    lat_ind = np.logical_and(mlat > extent[2], mlat < extent[3])

    # find i and j indices of lon/lat in boundaries
    ind = np.where(np.logical_and(lon_ind, lat_ind))

    # subset data from min i,j corner to max i,j corner
    # there will be some points outside of defined boundaries because grid is not rectangular
    sub = np.squeeze(ds)[range(np.min(ind[0]), np.max(ind[0]) + 1), range(np.min(ind[1]), np.max(ind[1]) + 1)]

    return sub


def wind_uv_to_dir(u, v):
    """
    Calculates the wind direction from the u and v component of wind.
    Takes into account the wind direction coordinates is different than the
    trig unit circle coordinate. If the wind direction is 360 then returns zero
    (by %360)
    Inputs:
    u = west/east direction (wind from the west is positive, from the east is negative)
    v = south/noth direction (wind from the south is positive, from the north is negative)
    """
    wdir = (270-np.rad2deg(np.arctan2(v, u))) % 360
    return wdir


def wind_uv_to_spd(u, v):
    """
    Calculates the wind speed from the u and v wind components
    :param u: west/east direction (wind from the west is positive, from the east is negative)
    :param v: south/noth direction (wind from the south is positive, from the north is negative)
    :returns WSPD: wind speed calculated from the u and v wind components
    """
    wspd = np.sqrt(np.square(u) + np.square(v))

    return wspd
