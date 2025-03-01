import json
import os
import pickle as pkl
from pathlib import PurePath
import re

import pandas as pd


with open("config.json") as file:
    config = json.load(file)
datapath = PurePath(
    os.path.expanduser("~"), config["intermediate_data"], "ship_routes"
)

imopath = PurePath(
    os.path.expanduser("~"), config["model_data"], "imo_by_type_2015.pkl"
)

with open(imopath, "rb") as file:
    ships_per_ship_class = pkl.load(file)


files = os.listdir(datapath)
speed_by_shiptype = {}
for file in files:
    df = pd.read_csv(
        PurePath(datapath, file),
        index_col=[0],
        parse_dates=True,
        dtype={"imo": int},
    )
    for k, v in ships_per_ship_class.items():
        t = df.loc[df["imo"].isin(v), :]
        t.loc[:, "group"] = pd.cut(
            t["speed_calc"],
            bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            right=False,
        )
        speed_by_shiptype[(int(re.findall("\d+", file)[0]), k)] = (
            t.groupby("group").count()["speed_calc"]
            / t.groupby("group").count()["speed_calc"].sum()
        )

speed_by_shiptype_df = pd.DataFrame(speed_by_shiptype).T
speed_by_shiptype_df.groupby(level=1).mean().to_csv(
    "time_average_speed_per_shiptype.csv"
)

active_ships_per_hour = {}
for file in files[0:4]:
    df = pd.read_csv(
        PurePath(datapath, file),
        index_col=[0],
        parse_dates=True,
        dtype={"imo": int},
    )
    df["hour"] = df.index.hour
    df = df.groupby(["hour", "imo"]).count().reset_index()
    for hour in range(0, 24):
        active_ships_per_hour[file, hour] = len(
            df[df.hour == hour].imo.unique()
        )
df.to_csv("active_ships_per_hour.csv")
