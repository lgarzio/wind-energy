#! /usr/bin/env python

import cartopy.crs as ccrs


ndbc = dict(
    NDBC44009=dict(
        lat=38.46,
        lon=-74.692,
        long_name='NDBC 44009',
        active='Current',
        text_offset=[.04, -0.02]
    ),
    NDBC44065=dict(
        lat=40.369,
        lon=-73.703,
        long_name='NDBC 44065',
        active='Current',
        text_offset=[.02, .02]
    ),
    NDBC44025=dict(
        lat=40.251,
        lon=-73.164,
        long_name='NDBC 44025',
        active='Current',
        text_offset=[.04, -0.02]
    )
)


projection_merc = dict(
    map=ccrs.Mercator(),
    data=ccrs.PlateCarree()
)

projection_lc = dict(
    map=ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8),
    data=ccrs.PlateCarree()
)

validation_points = dict(
    NYNE05=dict(
        lat=39.9686,
        lon=-72.7173,
        long_name='NYSERDA North Buoy',
        active='Archived',
        text_offset=[.02, .02]
    ),
    NYSE06=dict(
        lat=39.5465,
        lon=-73.4295,
        long_name='NYSERDA South Buoy',
        active='Archived',
        text_offset=[.02, .02]
    ),
    ASOSB6=dict(
        lat=39.2717,
        lon=-73.8892,
        long_name='Atlantic Shores Buoy 6',
        active='Archived',
        text_offset=[.02, .02]
    ),
    ASOSB4=dict(
        lat=39.2025,
        lon=-74.08194,
        long_name='Atlantic Shores Buoy 4',
        active='Current',
        text_offset=[.065, -.015]
    ),
    RUOYC=dict(
        lat=39.817005,
        lon=-74.213356,
        long_name='Oyster Creek',
        active='Archived',
        text_offset=[-.3, .02]
    ),
    SODAR=dict(
        lat=39.521396,
        lon=-74.319087,
        long_name='RUMFUS SODAR',
        active='Current',
        text_offset=[-.3, .04]
    ),
    NYSWE05=dict(
        lat=39.4847,
        lon=-73.5901,
        long_name='NYSERDA South West Buoy',
        active='Current',
        text_offset=[-.35, .03]
    )
)
