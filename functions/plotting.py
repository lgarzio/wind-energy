#! /usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 4/20/2021
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
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


# def add_lease_area_polygon_test(ax, lease_area_dict, line_color):
#     """
#     Adds polygon outlines for wind energy lease areas to map
#     :param ax: plotting axis object
#     :param lease_area_dict: dictionary containing lat/lon coordinates for wind energy lease area polygons
#     :param line_color: polygon line color
#     """
#     for key, value in lease_area_dict.items():
#         polygon = Polygon(value['outer'])
#         ax.plot(*polygon.exterior.xy, color=line_color, transform=ccrs.PlateCarree())


def add_lease_area_polygon(ax, lease_area_dict, line_color, lw=.8):
    """
    Adds polygon outlines for wind energy lease areas to map
    :param ax: plotting axis object
    :param lease_area_dict: dictionary containing lat/lon coordinates for wind energy lease area polygons
    :param line_color: polygon line color
    """
    for k, v in lease_area_dict.items():
        if len(v) > 0:
            for i, coord in enumerate(v):
                if i > 0:
                    poly_lons = [v[i - 1][0], coord[0]]
                    poly_lats = [v[i - 1][1], coord[1]]
                    ax.plot(poly_lons, poly_lats, ls='-', lw=lw, color=line_color, transform=ccrs.PlateCarree())

def add_lease_area_polygon_single(ax, lease_area_list, line_color, lw=.8):
    """
    Adds polygon outlines for wind energy lease areas to map
    :param ax: plotting axis object
    :param lease_area_list: list containing lat/lon coordinates for wind energy lease area polygons
    :param line_color: polygon line color
    """
    for i, coord in enumerate(lease_area_list):
        if i > 0:
            poly_lons = [lease_area_list[i - 1][0], coord[0]]
            poly_lats = [lease_area_list[i - 1][1], coord[1]]
            ax.plot(poly_lons, poly_lats, ls='-', lw=lw, color=line_color, transform=ccrs.PlateCarree())


def add_map_features(ax, axes_limits, xticks=None, yticks=None, landcolor=None, ecolor=None, zoom_shore=None):
    """
    Adds latitude and longitude gridlines and labels, coastlines, and statelines to a cartopy map object
    :param ax: plotting axis object
    :param axes_limits: list of axis limits [min lon, max lon, min lat, max lat]
    :param xticks: optional list of x tick locations
    :param yticks: optiona list of y tick locations
    :param landcolor: optional, specify land color
    :param ecolor: optional, specify edge color, default is black
    :param zoom_shore: optional, set to True if zooming into the shoreline (provides more resolution)
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

    if zoom_shore:
        land = cfeature.GSHHSFeature(scale='full')
    else:
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


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_contourf(fig, ax, x, y, c, cmap, levels=None, ttl=None, clab=None, cbar_ticks=None, extend=None,
                  shift_subplot_right=None, xlab=None, ylab=None, yticks=None):
    """
    Create a filled contour plot with user-defined levels and colors
    :param fig: figure object
    :param ax: plotting axis object
    :param x: x-axis data
    :param y: y-axis data
    :param c: color data
    :param cmap: colormap
    :param levels: optional list of data levels
    :param ttl: optional plot title
    :param clab: optional colorbar label
    :param cbar_ticks: optional, specify colorbar ticks
    :param extend: optional, different colorbar extensions, default is 'both'
    :param shift_subplot_right: optional, specify shifting the subplot, default is 0.88
    :param xlab: optional x-label
    :param ylab: optional y-label
    :param yticks: specify optional yticks
    :returns fig, ax objects
    """
    ttl = ttl or None
    levels = levels or None
    clab = clab or None
    cbar_ticks = cbar_ticks or None
    extend = extend or 'both'
    shift_subplot_right = shift_subplot_right or 0.88
    xlab = xlab or None
    ylab = ylab or None
    yticks = yticks or None

    plt.subplots_adjust(right=shift_subplot_right)
    if ttl:
        plt.title(ttl, fontsize=17)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    if levels:
        try:
            cs = ax.contourf(x, y, c, levels=levels, cmap=cmap, extend=extend, transform=ccrs.PlateCarree())
        except ValueError:
            cs = ax.contourf(x, y, c, levels=levels, cmap=cmap, extend=extend)
    else:
        try:
            cs = ax.contourf(x, y, c, cmap=cmap, extend=extend, transform=ccrs.PlateCarree())
        except ValueError:
            cs = ax.contourf(x, y, c, cmap=cmap, extend=extend)

    if cbar_ticks:
        cb = plt.colorbar(cs, cax=cax, ticks=cbar_ticks)
    else:
        cb = plt.colorbar(cs, cax=cax)

    if clab:
        cb.set_label(label=clab, fontsize=14)

    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if yticks:
        ax.set_yticks(yticks)

    return fig, ax


def plot_contourf_2leftaxes(fig, ax, x, y, y2, c, cmap, levels=None, ttl=None, clab=None, cbar_ticks=None, extend=None,
                  shift_subplot_right=None, shift_subplot_left=None, xlab=None, ylab=None, yticks=None):
    """
    Create a filled contour plot with user-defined levels and colors
    :param fig: figure object
    :param ax: plotting axis object
    :param x: x-axis data
    :param y: y-axis data
    :param c: color data
    :param cmap: colormap
    :param levels: optional list of data levels
    :param ttl: optional plot title
    :param clab: optional colorbar label
    :param cbar_ticks: optional, specify colorbar ticks
    :param extend: optional, different colorbar extensions, default is 'both'
    :param shift_subplot_right: optional, specify shifting the subplot, default is 0.88
    :param xlab: optional x-label
    :param ylab: optional y-label
    :param yticks: specify optional yticks
    :returns fig, ax objects
    """
    ttl = ttl or None
    levels = levels or None
    clab = clab or None
    cbar_ticks = cbar_ticks or None
    extend = extend or 'both'
    shift_subplot_right = shift_subplot_right or 0.88
    shift_subplot_left = shift_subplot_left or None
    xlab = xlab or None
    ylab = ylab or None
    yticks = yticks or None

    if shift_subplot_left:
        plt.subplots_adjust(right=shift_subplot_right, left=shift_subplot_left)
    else:
        plt.subplots_adjust(right=shift_subplot_right)
    if ttl:
        plt.title(ttl, fontsize=17)
    # divider = make_axes_locatable(ax)
    # cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    # fig.add_axes(cax)

    if levels:
        try:
            cs = ax.contourf(x, y, c, levels=levels, cmap=cmap, extend=extend, transform=ccrs.PlateCarree())
        except ValueError:
            cs = ax.contourf(x, y, c, levels=levels, cmap=cmap, extend=extend)
    else:
        try:
            cs = ax.contourf(x, y, c, cmap=cmap, extend=extend, transform=ccrs.PlateCarree())
        except ValueError:
            cs = ax.contourf(x, y, c, cmap=cmap, extend=extend)

    if cbar_ticks:
        cb = plt.colorbar(cs, ticks=cbar_ticks, pad=.02)
    else:
        cb = plt.colorbar(cs, pad=.02)

    if clab:
        cb.set_label(label=clab, fontsize=14)

    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if yticks:
        ax.set_yticks(yticks)

    # plot secondary y-axis data
    ax2 = ax.twinx()
    p2, = ax2.plot(x, y2, '--', c='gray', lw=.75)
    ax2.spines["left"].set_position(("axes", -0.15))
    make_patch_spines_invisible(ax2)
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_ylabel("Elevation (m)")
    ax2.yaxis.label.set_color(p2.get_color())
    tkw = dict(size=4, width=1.3)
    ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)

    return fig, ax


def plot_pcolormesh(fig, ax, x, y, c, var_lims=None, cmap=None, clab=None, ttl=None, extend=None,
                    shift_subplot_right=None, xlab=None, ylab=None, yticks=None, shading=None, norm_clevs=None,
                    cbar_ticks=None, title_size=None, cax_size=None):
    """
    Create a pseudocolor plot
    :param fig: figure object
    :param ax: plotting axis object
    :param x: x-axis data
    :param y: y-axis data
    :param c: color data
    :param var_lims: optional [min, max] values for plotting (for fixed colorbar)
    :param cmap: optional color map, default is jet
    :param clab: optionalcolorbar label
    :param ttl: optional plot title
    :param extend: optional, different colorbar extensions, default is 'both'
    :param shift_subplot_right: optional, specify shifting the subplot, default is 0.88
    :param xlab: optional x-label
    :param ylab: optional y-label
    :param yticks: specify optional yticks
    :param shading: optional shading ('auto', 'nearest', 'gouraud') default is 'auto'
    :param norm_clevs: optional normalized levels
    :param cbar_ticks: optional, specify colorbar ticks
    :param title_size: optional, specify title size, default 17
    :param cax_size: optional, color axis size, default 5%
    """
    var_lims = var_lims or None
    cmap = cmap or plt.get_cmap('jet')
    clab = clab or None
    ttl = ttl or None
    extend = extend or 'both'
    shift_subplot_right = shift_subplot_right or 0.88
    xlab = xlab or None
    ylab = ylab or None
    yticks = yticks or None
    shading = shading or 'auto'
    norm_clevs = norm_clevs or None
    cbar_ticks = cbar_ticks or None
    title_size = title_size or 17
    cax_size = cax_size or '5%'

    plt.subplots_adjust(right=shift_subplot_right)
    if ttl:
        plt.title(ttl, fontsize=title_size)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size=cax_size, pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    if var_lims:
        try:
            h = ax.pcolormesh(x, y, c, vmin=var_lims[0], vmax=var_lims[1], shading=shading, cmap=cmap,
                              transform=ccrs.PlateCarree())
        except ValueError:
            h = ax.pcolormesh(x, y, c, vmin=var_lims[0], vmax=var_lims[1], shading=shading, cmap=cmap)
    elif norm_clevs:
        try:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap, norm=norm_clevs, transform=ccrs.PlateCarree())
        except ValueError:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap, norm=norm_clevs)
    else:
        try:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap, transform=ccrs.PlateCarree())
        except ValueError:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap)

    if cbar_ticks:
        cb = plt.colorbar(h, cax=cax, extend=extend, ticks=cbar_ticks)
    else:
        cb = plt.colorbar(h, cax=cax, extend=extend)

    if clab:
        cb.set_label(label=clab, fontsize=12)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if yticks:
        ax.set_yticks(yticks)


def plot_pcolormesh_2leftaxes(fig, ax, x, y, y2, c, var_lims=None, cmap=None, clab=None, ttl=None, extend=None,
                    shift_subplot_right=None, shift_subplot_left=None, xlab=None, ylab=None, yticks=None, shading=None, norm_clevs=None,
                    cbar_ticks=None):
    """
    Create a pseudocolor plot
    :param fig: figure object
    :param ax: plotting axis object
    :param x: x-axis data
    :param y: y-axis data
    :param c: color data
    :param var_lims: optional [min, max] values for plotting (for fixed colorbar)
    :param cmap: optional color map, default is jet
    :param clab: optionalcolorbar label
    :param ttl: optional plot title
    :param extend: optional, different colorbar extensions, default is 'both'
    :param shift_subplot_right: optional, specify shifting the subplot, default is 0.88
    :param xlab: optional x-label
    :param ylab: optional y-label
    :param yticks: specify optional yticks
    :param shading: optional shading ('auto', 'nearest', 'gouraud') default is 'auto'
    :param norm_clevs: optional normalized levels
    :param cbar_ticks: optional, specify colorbar ticks

    """
    var_lims = var_lims or None
    cmap = cmap or plt.get_cmap('jet')
    clab = clab or None
    ttl = ttl or None
    extend = extend or 'both'
    shift_subplot_right = shift_subplot_right or 0.88
    shift_subplot_left = shift_subplot_left or None
    xlab = xlab or None
    ylab = ylab or None
    yticks = yticks or None
    shading = shading or 'auto'
    norm_clevs = norm_clevs or None
    cbar_ticks = cbar_ticks or None

    if shift_subplot_left:
        plt.subplots_adjust(right=shift_subplot_right, left=shift_subplot_left)
    else:
        plt.subplots_adjust(right=shift_subplot_right)
    if ttl:
        plt.title(ttl, fontsize=17)
    # divider = make_axes_locatable(ax)
    # cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    # fig.add_axes(cax)

    if var_lims:
        try:
            h = ax.pcolormesh(x, y, c, vmin=var_lims[0], vmax=var_lims[1], shading=shading, cmap=cmap,
                              transform=ccrs.PlateCarree())
        except ValueError:
            h = ax.pcolormesh(x, y, c, vmin=var_lims[0], vmax=var_lims[1], shading=shading, cmap=cmap)
    elif norm_clevs:
        try:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap, norm=norm_clevs, transform=ccrs.PlateCarree())
        except ValueError:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap, norm=norm_clevs)
    else:
        try:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap, transform=ccrs.PlateCarree())
        except ValueError:
            h = ax.pcolormesh(x, y, c, shading=shading, cmap=cmap)

    if cbar_ticks:
        cb = plt.colorbar(h, extend=extend, ticks=cbar_ticks, pad=.02)
    else:
        cb = plt.colorbar(h, extend=extend, pad=.02)

    if clab:
        cb.set_label(label=clab, fontsize=14)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if yticks:
        ax.set_yticks(yticks)

    # plot secondary y-axis data
    ax2 = ax.twinx()
    p2, = ax2.plot(x, y2, '--', c='gray', lw=.75)
    ax2.spines["left"].set_position(("axes", -0.15))
    make_patch_spines_invisible(ax2)
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_ylabel("Elevation (m)")
    ax2.yaxis.label.set_color(p2.get_color())
    tkw = dict(size=4, width=1.3)
    ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)


def plot_windrose(axis, wspd, wdir, ttl):
    # set the bins for wind speeds
    b = [0, 5, 10, 15, 20, 25, 30]
    axis.bar(wdir, wspd, normed=True, bins=b, opening=1, edgecolor='black', cmap=cm.jet, nsector=36)

    # add % to y-axis labels
    newticks = ['{:.0%}'.format(x / 100) for x in axis.get_yticks()]
    axis.set_yticklabels(newticks)

    # format legend
    # move legend
    al = axis.legend(borderaxespad=-7, title='Wind Speed (m/s)')

    # replace the text in the legend
    text_str = ['0$\leq$ ws <5', '5$\leq$ ws <10', '10$\leq$ ws <15', '15$\leq$ ws <20', '20$\leq$ ws <25',
                '25$\leq$ ws <30', 'ws $\geq$30']
    for i, txt in enumerate(al.get_texts()):
        txt.set_text(text_str[i])
    plt.setp(al.get_texts(), fontsize=10)

    # add title
    axis.set_title(ttl, fontsize=14)


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
