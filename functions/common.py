#! /usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 4/13/2021
"""

import numpy as np
import xml.etree.ElementTree as ET  # for parsing kml files


def extract_lease_areas():
    """
    Extracts polygon coordinates from a .kml file.
    :returns dictionary containing lat/lon coordinates for wind energy lease area polygons, separated by company
    """
    # boem_lease_areas = '/Users/garzio/Documents/rucool/bpu/wrf/boem_lease_area_full.kml'  # on local machine
    # boem_lease_areas = '/Users/garzio/Documents/rucool/bpu/wrf/boem_lease_areas_AS_OW_split.kml'  # on local machine
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


def subset_grid(data, model):
    """
    Subset the data according to defined latitudes and longitudes, and define the axis limits for the plots
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :param model: the model version that is being plotted, e.g. 3km, 9km, or bight (to plot just NY Bight region)
    :returns data: data subset to the desired grid region
    :returns axis_limits: axis limits to be used in the plotting function
    """
    if model == '3km':
        axis_limits = [-79.79, -69.2, 34.5, 43]
        model_lims = dict(minlon=-79.9, maxlon=-69, minlat=34.5, maxlat=43)
    elif model == '9km':
        axis_limits = [-80, -67.9, 33.05, 44]
        model_lims = dict(minlon=-80.05, maxlon=-67.9, minlat=33, maxlat=44.05)
    elif model == 'bight':
        axis_limits = [-77.5, -72, 37.5, 42.05]
        model_lims = dict(minlon=-77.55, maxlon=-71.95, minlat=37.45, maxlat=42.05)
    else:
        print('Model not recognized')

    mlon = data['XLONG']
    mlat = data['XLAT']
    lon_ind = np.logical_and(mlon > model_lims['minlon'], mlon < model_lims['maxlon'])
    lat_ind = np.logical_and(mlat > model_lims['minlat'], mlat < model_lims['maxlat'])

    # find i and j indices of lon/lat in boundaries
    ind = np.where(np.logical_and(lon_ind, lat_ind))

    # subset data from min i,j corner to max i,j corner
    # there will be some points outside of defined boundaries because grid is not rectangular
    data = np.squeeze(data)[:, range(np.min(ind[0]), np.max(ind[0]) + 1), range(np.min(ind[1]), np.max(ind[1]) + 1)]

    return data, axis_limits


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
