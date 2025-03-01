import os
import json

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import logging
import matplotlib.pyplot as plt

from datacube.utils import geometry

import pyproj

from shapely.geometry import (
    box,
    mapping,
)
from shapely.ops import transform

from rasterio.enums import MergeAlg
from rioxarray.rioxarray import affine_to_coords
from rasterio.features import rasterize

# set cache for gdal to 3 times of default (which is 5% of max memory)
import gdal

default_cache_max = gdal.GetCacheMax()
gdal.SetCacheMax(int(3 * default_cache_max))

logger = logging.getLogger(__name__)
# from geocube.api.core import make_geocube
# from geocube.rasterize import rasterize_image
# from functools import partial
# from logger import logger


def get_lcc_bounds(bounding_box, input_crs="epsg:4326", to_shp=False):
    """
    Usage
    ------

    with open("config.json") as file:
        config = json.load(file)
    bbox = config["bounding_box_lcc"]
    get_lcc_bounds(
            box(bbox[0], bbox[1], bbox[2], bbox[3]),
            input_crs="+proj=lcc +lat_1=30 +lat_2=60 +lat_0=55 +lon_0=10 +y_0=0 +x_0=0 +a=6370997 +b=6370997 +units=m +no_defs",
            to_shp=True)

    """
    lon, lat = bounding_box.exterior.coords.xy
    geodf = gpd.GeoDataFrame(
        crs=input_crs, geometry=gpd.points_from_xy(lon, lat),
    )
    geodf.drop(4, inplace=True)
    geodf.index = ["lower right", "upper right", "upper left", "lower left"]

    geodf = geodf.to_crs(
        "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=55 +lon_0=10 +y_0=0 +x_0=0 +a=6370997 +b=6370997 +units=m +no_defs"
    )
    if to_shp:
        geodf.to_file(driver="ESRI Shapefile", filename="area.shp")
    return geodf


def plot_array(array, lower=0, upper=3000):
    """
    """
    fig, ax = plt.subplots()
    im = ax.imshow(pd.DataFrame(array).clip(lower=lower, upper=upper).values)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Emission", rotation=-90, va="bottom")


def rasterize_points(
    config=None,
    emission_types={"CO2 [kg]": "CO2", "NOx [kg]": "NOx",},
    # resolution=(-0.03, 0.05),
    # bbox=[-4, 50, 25, 65],
    # crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=55 +lon_0=10 +y_0=0 +x_0=0 +a=6370997 +b=6370997 +units=m +no_defs"
    crs="EPSG:4326",
):
    """
    Parameter
    ----------

    crs: str
        CRS System of the input bounding box. For LCC use:
        "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=55 +lon_0=10 +y_0=0 +x_0=0 +a=6370997 +b=6370997 +units=m +no_defs"

    """

    if config is None:
        logging.info("No config file provided, trying to read...")
        with open("config.json") as file:
            config = json.load(file)

    # get resolution in lon/lat because rasterizing will be in epsg 4326
    resolution = config["resolution_lonlat"]
    # get the square box in LCC coordinates
    if crs == "EPSG:4326":
        bbox = config["bounding_box_cmaq"]  # lonlat
    else:
        bbox = config["bounding_box_lcc"]

    datapath = os.path.join(
        os.path.expanduser("~"),
        config["intermediate_data"],
        config["scenario"],
        "ship_emissions",
    )

    filepaths = [os.path.join(datapath, i,) for i in os.listdir(datapath)]
    filepaths.sort()

    # path to store data
    result_data = os.path.join(
        os.path.expanduser("~"), config["result_data"], config["scenario"]
    )

    if not os.path.exists(result_data):
        os.makedirs(result_data)

    out_crs = "EPSG:4326"
    bounding_box = box(bbox[0], bbox[1], bbox[2], bbox[3])

    # reproject LCC to epsg:4326 i.e. lon/lat
    project = pyproj.Transformer.from_crs(
        pyproj.CRS(crs), pyproj.CRS(out_crs), always_xy=True
    ).transform
    bounding_box = transform(project, bounding_box)

    json_box = mapping(bounding_box)  # minx miny maxx maxy

    json_box["crs"] = {"properties": {"name": out_crs}}

    geopoly = geometry.Geometry(json_box, crs=crs,)
    geobox = geometry.GeoBox.from_geopolygon(
        geopoly, resolution, crs=crs,
    )  # resolution y,x

    # geobox.xr_coords() # also get coords as xarrays from geobox
    coords = affine_to_coords(geobox.affine, geobox.width, geobox.height)

    for emission_type in emission_types.keys():
        logging.info(
            "Rasterizing emissions for type: {}.".format(emission_type)
        )
        emissions_per_hour = {}
        timestamps = []
        filepaths.sort()
        for file in filepaths:
            # print("Do file: {}".format(file))
            # select only certain day
            if "20" in file:
                df_day = pd.read_csv(
                    file,
                    index_col="datetime",
                    parse_dates=True,
                    compression="zip",
                )  # , nrows=1000000)

                # add both engine types
                df_day[emission_type] = (
                    df_day["Propulsion-" + emission_type]
                    + df_day["Electrical-" + emission_type]
                )
                for hour in range(0, 24):
                    df_hour = df_day[df_day.index.hour == hour]

                    geodf = gpd.GeoDataFrame(
                        df_hour,
                        crs="epsg:4326",
                        geometry=gpd.points_from_xy(df_hour.lon, df_hour.lat),
                    )

                    logging.info(
                        "Rasterzing day {} and hour {}".format(
                            df_day.index[0].dayofyear, hour
                        )
                    )
                    arr = rasterize(
                        zip(
                            geodf.geometry.apply(mapping).values,
                            geodf[emission_type],
                        ),  # colums 7 is co2
                        out_shape=(geobox.height, geobox.width,),
                        transform=geobox.affine,
                        merge_alg=MergeAlg.add,
                        all_touched=True,
                    )
                    # hour = (df_day.index[
                    #     0
                    # ].dayofyear - 1) * 24 + hour # -1 to start with 0
                    timestamp = df_hour.index[0].strftime("%Y-%m-%d-%H")
                    timestamps.append(timestamp)
                    emissions_per_hour[timestamp] = arr
            else:
                logging.info("Filepath {} skipped, due to if.".format(file))

        if timestamps == sorted(timestamps, reverse=True):
            logging.error("Order of dates is not ascending!")

        # import pdb; pdb.set_trace()
        da = xr.DataArray(
            [i for i in emissions_per_hour.values()],
            dims=["time", "lat", "lon",],
            coords=[
                np.array(range(0, len(timestamps))),
                coords["y"],
                coords["x"],
            ],
        )

        da = da.rename("sum")
        da = da.astype("float32")
        da.attrs = {"units": "kg h-1"}

        da.coords["time"].attrs = {
            "standard_name": "time",
            "calendar": "proleptic_gregorian",
            "units": "Hours since 2014-12-01",
            "axis": "T",
        }
        da.coords["lon"].attrs = {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
        da.coords["lat"].attrs = {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
        da.to_netcdf(
            os.path.join(
                result_data, emission_types[emission_type] + ".nc"
            ),  # write to shorter file name
            encoding={
                "lat": {"dtype": "float32"},
                "lon": {"dtype": "float32"},
                "sum": {"dtype": "float32"},
            },
        )


# rasterize_points()
