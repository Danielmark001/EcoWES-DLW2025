# import folium
# from folium import plugins
import geopandas as gpd
import pandas as pd
import os

# import contextily as cx
import matplotlib.pyplot as plt
import xarray as xr
import json
import matplotlib as mpl
from matplotlib.patches import Rectangle

with open("config.json") as file:
    config = json.load(file)


nc_dirpath = os.path.join(
    os.path.expanduser("~"), config["result_data"], "2015_sq"
)
ds = []
for file in os.listdir(nc_dirpath):
    nc_filepath = os.path.join(nc_dirpath, file)
    ds.append(xr.open_dataset(nc_filepath).rename({"sum": file.strip(".nc")}))

pollutants = xr.merge(ds)
# select = ds.sum("time")["sum"]
# ds["sum"].isel(time=1)
pollutant = "CO2"
select = pollutants.sum("time")[pollutant]
# select = select / 1e6 # kg -> Gg
fig, ax = plt.subplots()
select.plot(
    ax=ax,
    cmap=mpl.cm.RdYlBu_r,
    norm=mpl.colors.LogNorm(
        vmin=float(select.min()) + 100, vmax=float(select.max())
    ),
    cbar_kwargs={
        # "orientation": "horizontal",
        "label": "CO2 emissions in kg",
        # "pad": 0.1,
    },
)
ax.add_patch(Rectangle((-5, 48), 20, 20, fill=False, color="red"))
plt.text(-3.5, 64, "Vesselfinder", color="red")
plt.text(20, 52, "Helcom", color="red")
plt.savefig("figures/{}-emissions.pdf".format(pollutant))


monthsum = pollutants.CO2_big.resample(time="1M").sum()


janjun = monthsum[[1, 7]]
# fig, ax = plt.subplots()
ax = janjun.plot(
    x="lon",
    y="lat",
    col="time",
    cmap=mpl.cm.RdYlBu_r,
    norm=mpl.colors.LogNorm(
        vmin=float(janjun.min()) + 100, vmax=float(janjun.max())
    ),
    cbar_kwargs={
        # "orientation": "horizontal",
        "label": "CO2 emissions in kg",
        # "pad": 0.1,
    },
)
month = {0: "January", 1: "July"}
for i, ax in enumerate(ax.axes.flat):
    ax.set_title(month[i])
# ax.add_patch(Rectangle((-5, 48), 20, 20, fill=False, color="red"))
# plt.title("June")
plt.savefig("figures/monthly-emissions-janjul-raster.pdf")
# plt.text(-3.5, 64, "Vesselfinder", color="red")
# plt.text(20, 52, "Helcom", color="red")

min_lon = 6
min_lat = 53
max_lon = 13
max_lat = 58

mask_lon = (monthsum.lon >= min_lon) & (monthsum.lon <= max_lon)
mask_lat = (monthsum.lat >= min_lat) & (monthsum.lat <= max_lat)
monthsum_sh = monthsum.where(mask_lon & mask_lat, drop=True)

janjun = monthsum_sh[[1, 7]]
# fig, ax = plt.subplots()
ax = janjun.plot(
    x="lon",
    y="lat",
    col="time",
    cmap=mpl.cm.RdYlBu_r,
    norm=mpl.colors.LogNorm(
        vmin=float(janjun.min()) + 100, vmax=float(janjun.max())
    ),
    cbar_kwargs={
        # "orientation": "horizontal",
        "label": "CO2 emissions in kg",
        # "pad": 0.1,
    },
)
month = {0: "January", 1: "July"}
for i, ax in enumerate(ax.axes.flat):
    ax.set_title(month[i])
plt.savefig("figures/monthly-emissions-janjul-raster_sh.pdf")
