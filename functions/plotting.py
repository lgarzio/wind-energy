#! /usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 4/13/2021
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_contours(ax, londata, latdata, vardata, clist, label_format=None):
    """
    Adds black contour lines with labels to a cartopy map object
    :param ax: plotting axis object
    :param londata: longitude data
    :param latdata: latitude data
    :param vardata: variable data
    :param clist: list of contour levels
    :param label_format: optional format for contour labels (e.g. '%.1f')
    """
    CS = ax.contour(londata, latdata, vardata, clist, colors='black', linewidths=.5, transform=ccrs.PlateCarree())
    if label_format is None:
        lf = '%d'
    else:
        lf = label_format
    ax.clabel(CS, inline=True, fontsize=10.5, fmt=lf)


def add_lease_area_polygon(ax, lease_area_dict, line_color):
    """
    Adds polygon outlines for wind energy lease areas to map
    :param ax: plotting axis object
    :param lease_area_dict: dictionary containing lat/lon coordinates for wind energy lease area polygons
    :param line_color: polygon line color
    """
    for key, value in lease_area_dict.items():
        for k, v in value.items():
            if len(v) > 0:
                for i, coord in enumerate(v):
                    if i > 0:
                        poly_lons = [v[i - 1][0], coord[0]]
                        poly_lats = [v[i - 1][1], coord[1]]
                        ax.plot(poly_lons, poly_lats, ls='-', lw=.4, color=line_color, transform=ccrs.PlateCarree())


def add_map_features(ax, axes_limits, xticks=None, yticks=None, landcolor=None, ecolor=None):
    """
    Adds latitude and longitude gridlines and labels, coastlines, and statelines to a cartopy map object
    :param ax: plotting axis object
    :param axes_limits: list of axis limits [min lon, max lon, min lat, max lat]
    :param xticks: optional list of x tick locations
    :param yticks: optiona list of y tick locations
    :param landcolor: optional, specify land color
    :param ecolor: optional, specify edge color, default is black
    """
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='dotted', x_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 13}
    gl.ylabel_style = {'size': 13}

    # add some space between the grid labels and bottom of plot
    gl.xpadding = 12
    gl.ypadding = 12

    gl.rotate_labels = False

    # set axis limits
    ax.set_extent(axes_limits)

    # set optional x and y ticks
    if xticks:
        gl.xlocator = mticker.FixedLocator(xticks)

    if yticks:
        gl.ylocator = mticker.FixedLocator(yticks)

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m')

    if landcolor is not None:
        lc = landcolor
    else:
        lc = 'none'

    if ecolor is not None:
        ec = ecolor
    else:
        ec = 'black'

    ax.add_feature(land, zorder=5, edgecolor=ec, facecolor=lc)

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

    ax.add_feature(cfeature.BORDERS, zorder=6, edgecolor=ec)
    ax.add_feature(state_lines, zorder=7, edgecolor=ec)


def plot_contourf(fig, ax, ttl, lon_data, lat_data, var_data, clevs, cmap, clab, var_min, var_max, normalize,
                  cbar_ticks=None):
    """
    Create a filled contour plot with user-defined levels and colors
    :param fig: figure object
    :param ax: plotting axis object
    :param ttl: plot title
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param clevs: list of colorbar level demarcations
    :param cmap: colormap
    :param clab: colorbar label
    :param var_min: optional, minimum value for plotting (for fixed colorbar)
    :param var_max: optional, maximum value for plotting (for fixed colorbar)
    :param normalize: optional, object that normalizes the colorbar level demarcations
    :param cbar_ticks: optional, specify colorbar ticks
    :returns fig, ax objects
    """
    plt.subplots_adjust(right=0.88)
    plt.title(ttl, fontsize=17)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    if normalize == 'yes':
        #norm = mpl.colors.BoundaryNorm(clevs, 15)
        norm = mpl.colors.BoundaryNorm(clevs, len(clevs)-1)
        cs = ax.contourf(lon_data, lat_data, var_data, clevs, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                         alpha=.9)
    else:
        cs = ax.contourf(lon_data, lat_data, var_data, clevs, vmin=var_min, vmax=var_max, cmap=cmap,
                         transform=ccrs.PlateCarree(), alpha=.9)

    if cbar_ticks is not None:
        cb = plt.colorbar(cs, cax=cax, ticks=cbar_ticks)
    else:
        cb = plt.colorbar(cs, cax=cax)

    cb.set_label(label=clab, fontsize=14)

    return fig, ax


def plot_pcolormesh(fig, ax, ttl, lon_data, lat_data, var_data, var_min, var_max, cmap, clab):
    """
    Create a pseudocolor plot
    :param fig: figure object
    :param ax: plotting axis object
    :param ttl: plot title
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param var_min: minimum value for plotting (for fixed colorbar)
    :param var_max: maximum value for plotting (for fixed colorbar)
    :param cmap: color map
    :param clab: colorbar label
    """
    plt.subplots_adjust(right=0.88)
    plt.title(ttl, fontsize=17)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    h = ax.pcolormesh(lon_data, lat_data, var_data, shading='gouraud', cmap=cmap, transform=ccrs.PlateCarree())
    #h = ax.pcolormesh(lon_data, lat_data, var_data, vmin=var_min, vmax=var_max, shading='gouraud', cmap=cmap, transform=ccrs.PlateCarree())

    cb = plt.colorbar(h, cax=cax)
    cb.set_label(label=clab, fontsize=14)


def set_map(data):
    """
    Set up the map and projection
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :returns fig, ax objects
    :returns dlat: latitude data values
    returns dlon: longitude data values
    """
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))

    dlat = data['XLAT'].values
    dlon = data['XLONG'].values

    return fig, ax, dlat, dlon