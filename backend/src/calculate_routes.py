import functools
import os
import logging

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Point
import geopandas

from helpers import haversine
from sklearn.metrics import mean_squared_error


def create_ship_route(temp_df, drift_speed=0.5, resample="5min"):
    """ Creates ship route from raw data

    Parameters
    ----------
    temp_df: pd.DataFrame()
        Dataframe with data of one ship

    Returns
    --------
    temp_df: pd.DataFrame()
        Processed dataframe with 5min ship routes
    """

    # sort values by time
    temp_df = temp_df.sort_values(by="datetime", ascending=True)

    # calculate distance between two points of  consecutive timesteps in m
    temp_df["dist"] = (
        haversine(
            temp_df.lat.shift().values,
            temp_df.lon.shift().values,
            temp_df["lat"].values,
            temp_df["lon"].values,
        )
        * 1000
    )

    # get time in seconds
    temp_df["tdiff"] = temp_df["datetime"].diff().dt.total_seconds()

    # calculate avg. speed based on distance an time-diff in m/s
    temp_df["speed_calc"] = temp_df["dist"] / temp_df["tdiff"]

    # temp_df.to_csv("raw.csv", mode="a", header=False)

    # temp_df["dist"] = temp_df["dist"].shift(-1)
    temp_df.dropna(
        inplace=True, subset=["speed_calc", "tdiff", "dist", "lon", "imo"]
    )

    # set date index
    temp_df = temp_df.set_index("datetime")

    # x = temp_df[(temp_df["tdiff"] > 3600 * 12)]

    # resample to 5min data
    temp_df = temp_df.resample(str(resample) + "min").mean()

    # temp_df.to_csv("resampled.csv", mode="a", header=False)

    temp_df["tdiff"] = temp_df["tdiff"].fillna(method="ffill")

    x = temp_df[
        (temp_df["tdiff"] > 3600 * 48)  # maximum 48 hours to interpolate
        | (
            (temp_df["dist"] > 300)  # or 300 m at the outer area
            & (temp_df["lon"] < -4.9)  # clip geo-bounds
        )
    ]

    for k in x.index:
        temp_df.drop(
            temp_df.loc[
                k - DateOffset(seconds=x.loc[k]["tdiff"] - (5 * 60)) : k
            ].index,
            inplace=True,
        )

    # temp_df.to_csv("clean.csv", mode="a",header=False)
    # interpolate lon/lat to get the positions of the ships
    temp_df[["lon", "lat"]] = temp_df[["lon", "lat"]].interpolate()

    # temp_df.to_csv("interpolate.csv", mode="a", header=False)

    # create speed for resample data
    temp_df[["speed_calc", "imo"]] = temp_df[["speed_calc", "imo"]].fillna(
        method="ffill"
    )

    temp_df["speed_calc"] = temp_df["speed_calc"].apply(
        lambda x: np.where(x < drift_speed, 0, x)
    )

    # sometimes there seems to a an error in lon/lat which causes
    # high distances at short time => very high speed, remove these values here
    # or set to zero, to keep intact X-min timeindex!?
    temp_df = temp_df[temp_df["speed_calc"] <= 15]

    # temp_df.to_csv("filled.csv", mode="a",header=False)

    return temp_df


def calculate_routes(config):
    """ Calculate the ship routes
    """
    # data path with original intput data
    datapath = os.path.join(os.path.expanduser("~"), config["merged"])

    # path to store data
    intermediate_path = os.path.join(
        os.path.expanduser("~"), config["intermediate_data"], "ship_routes"
    )
    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path)

    files = os.listdir(datapath)

    logging.info(
        "Start looping over all raw-data files in {} to generate routes.".format(
            datapath
        )
    )
    for file in files:
        path = os.path.join(datapath, file)
        logging.info("Read preprocessed file {}.".format(file))
        df = pd.read_csv(
            path,
            dtype={"imo": np.int32},
            engine="c",
            # nrows=10000,
        )

        df["datetime"] = pd.to_datetime(df["datetime"])

        imo_numbers = df["imo"].unique()
        logging.info(
            "Unique IMO numbers in raw data file are: {}.".format(
                len(imo_numbers)
            )
        )

        # ship_routes = pd.DataFrame()
        if not os.path.exists(intermediate_path):
            os.makedirs(intermediate_path)
        outputpath = os.path.join(intermediate_path, "ship_routes_" + file)

        ship_routes = []
        logging.info(
            "Loop over ships by IMO number and writing results to: {}.".format(
                outputpath
            )
        )
        for i in imo_numbers:
            temp_df = df[df["imo"] == i]
            ship_route = create_ship_route(
                temp_df,
                drift_speed=config["drift_speed"],
                resample=config["resample"],
            )
            ship_route = ship_route.dropna(subset=["speed_calc"])
            ship_routes.append(ship_route)

        ship_routes_df = pd.concat(ship_routes)
        ship_routes_df.to_csv(outputpath)
        logging.info(
            "Unique IMO numbers in routes are: {}".format(
                len(ship_routes_df["imo"].unique())
            )
        )
        # ship_routes = pd.concat(ship_routes)
        # ship_routes = ship_routes.reset_index()
        # ship_routes.drop("datetime", axis=1, inplace=True)
        # ship_routes["geometry"] = ship_routes.apply(
        #     lambda x: Point((float(x.lon), float(x.lat))), axis=1
        # )
        #
        # geodf = geopandas.GeoDataFrame(ship_routes, geometry="geometry")
        # # proj WGS84
        # geodf.crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        # geodf.to_file("data/100-raw.shp", driver="ESRI Shapefile")

        # number_nans = ship_routes["speed_calc"].isna().sum()
        # ship_routes = ship_routes.dropna(subset=["speed_calc"])
        # logging.info("Dropped {} rows with NaN in shiproutes.".format(number_nans))

        # logging.info("Calculate RMS for speed.")
        # rms = np.sqrt(
        #     mean_squared_error(ship_routes[["speed"]].fillna(0), ship_routes[["speed_calc"]])
        # )
        # logging.info("RMS error of speed is {}".format(rms))


if __name__ == "__main__":

    import json
    from logger import logger

    logging.info("Start data processing...")

    with open("config.json") as file:
        config = json.load(file)

    calculate_routes(config)
