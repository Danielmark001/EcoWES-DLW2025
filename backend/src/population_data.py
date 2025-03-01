import os
import json

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import logging
import matplotlib.pyplot as plt


from shapely.geometry import (
    box,
    mapping,
    Point,
)
import pyproj

from shapely.ops import (
    cascaded_union,
    transform,
)
from rasterio.enums import MergeAlg
from rioxarray.rioxarray import affine_to_coords
from rasterio.features import rasterize

from datacube.utils import geometry

# get schleswig holstein polygon (inkl. water)
basepath = "/home/admin/nextcloud-znes/KlimaSchiff/population_data"
sh_shapes = gpd.read_file(
    os.path.join(basepath, "shapefile_sh", "schleswig-holstein.shp",),
    crs="4326",
)
polygons = sh_shapes.geometry


# Read ZENSUS data
df = pd.read_csv(
    os.path.join(
        basepath, "csv_Demographie_100m_Gitter/Bevoelkerung100M.csv",
    ),
    sep=";",
    encoding="latin_1",
)
df_age = df[(df["Merkmal"] == "ALTER_10JG")]
df_age = df_age.set_index("Gitter_ID_100m")
df_age.index.name = "id"
df_age = df_age.drop(["Gitter_ID_100m_neu", "Merkmal",], axis=1,)
df_age = df_age.set_index("Auspraegung_Code", append=True,)

df_by_age = df_age["Anzahl"].unstack()

# get INSPIRE conform grid data 100mx100m (only select north boxes N35, N34, N33)
grid_path = os.path.join(basepath, "DE_Grid_ETRS89_LAEA_100m/geogitter",)

files = os.listdir(grid_path)
df_grid = pd.DataFrame()
for file in files:
    if "N35" in file or "N34" in file or "N33" in file:
        df_grid = pd.concat(
            [
                df_grid,
                pd.read_csv(
                    os.path.join(grid_path, file,), sep=";", header=None,
                ),
            ]
        )
df_grid.columns = [
    "id",
    "x_sw",
    "y_sw",
    "x_mp",
    "y_mp",
    "f_staat",
    "f_land",
    "f_wasser",
    "p_staat",
    "p_land",
    "p_wasser",
    "ags",
]
df_grid = df_grid.set_index("id")

# Combine grid with population data
df_pop = df_grid[["x_mp", "y_mp",]].merge(
    df_by_age, how="inner", left_index=True, right_index=True,
)

# convert geometry to 4326
crs = "epsg:4326"

geodf_pop = gpd.GeoDataFrame(
    df_pop,
    crs="epsg:3035",
    geometry=gpd.points_from_xy(df_pop.x_mp, df_pop.y_mp,),
)
geodf_pop = geodf_pop.to_crs(crs)

# clip geomtry by SH boundaries
sh_poly = cascaded_union(polygons.to_crs(crs))
geodf_sh = gpd.clip(geodf_pop, sh_poly,)

with open("config.json") as file:
    config = json.load(file)

resolution = config["resolution"]

bbox = config["bounding_box"]

bounding_box = box(bbox[0], bbox[1], bbox[2], bbox[3],)

# if crs == "epsg:3035":
#     project = pyproj.Transformer.from_crs(
#         pyproj.CRS('EPSG:4326'),
#         pyproj.CRS('EPSG:3035'), always_xy=True).transform
#     bounding_box = transform(project, bounding_box)
#     wgs84_pt = Point(resolution)
#     resolution = [1000, 1000]

json_box = mapping(bounding_box)  # minx miny maxx maxy

json_box["crs"] = {"properties": {"name": crs}}

geopoly = geometry.Geometry(json_box, crs=crs,)
geobox = geometry.GeoBox.from_geopolygon(
    geopoly, resolution, crs=crs,
)  # resolution y,x

# geobox.xr_coords() # also get coords as xarrays from geobox
coords = affine_to_coords(geobox.affine, geobox.width, geobox.height,)

geodf_sh = geodf_sh.fillna(
    0
)  # need to replace NaN for rasterize to work properly
pop_by_ageclass = {}
for age in geodf_sh.columns[2:11]:
    arr = rasterize(
        zip(geodf_sh.geometry.apply(mapping).values, geodf_sh[age],),
        out_shape=(geobox.height, geobox.width,),
        transform=geobox.affine,
        merge_alg=MergeAlg.add,
        default_value=np.NaN,
        # fill=1e6,
        all_touched=True,
    )
    da = xr.DataArray(
        arr,
        name=str(age),
        dims=["lat", "lon",],
        coords=[coords["y"], coords["x"],],
    )

    da = da.where(da > 0)  # replace 0 by nan by masking

    pop_by_ageclass[age] = da

ds = xr.combine_by_coords([i.to_dataset() for i in pop_by_ageclass.values()])

ds.to_netcdf(
    os.path.join(
        "/home/admin/nextcloud-znes/KlimaSchiff/population_data",
        "population.nc",
    ),  # write to shorter file name
    encoding={"lat": {"dtype": "float32"}, "lon": {"dtype": "float32"},},
)

# fig, ax = plt.subplots()
# im = ax.imshow(pd.DataFrame(pop_by_ageclass[2].values).clip(lower=0, upper=1000).values)
# cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel("Population", rotation=-90, va="bottom")
