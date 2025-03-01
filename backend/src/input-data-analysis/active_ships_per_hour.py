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

active_ships_per_hour = {}
for file in files:
    df = pd.read_csv(
        PurePath(datapath, file),
        index_col=[0],
        parse_dates=True,
        dtype={"imo": int},
    )
    df["hour"] = df.index.hour
    df = df.groupby(["hour", "imo"]).count().reset_index()
    for hour in range(0, 24):
        active_ships_per_hour[int(re.sub("[^0-9]", "", file)), hour] = len(
            df[df.hour == hour].imo.unique()
        )
pd.Series(active_ships_per_hour).to_csv("active_ships_per_hour.csv")
