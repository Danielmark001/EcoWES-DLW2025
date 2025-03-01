"""
Get set of unique imo numbers from ship routes
"""
import os
import json
import pickle
import pandas as pd

from itertools import chain


def calc_unique_imos():
    """
    """
    with open("config.json") as file:
        config = json.load(file)

    dirpath = os.path.join(
        os.path.expanduser("~"), config["intermediate_data"], "ship_routes"
    )

    files = os.listdir(dirpath)

    imos = []
    for file in files:
        if file.endswith(".csv"):
            # print(file)
            _df = pd.read_csv(
                os.path.join(dirpath, file), usecols=["imo"], dtype="int"
            )
            imos.append(_df.imo.unique())

    imos = list(chain.from_iterable(imos))

    outpath = os.path.join(os.path.expanduser("~"), config["model_data"])

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    unique_imos = set(imos)

    pd.Series(list(unique_imos)).to_csv(
        os.path.join(
            os.path.expanduser("~"), config["model_data"], "unique_imos.csv"
        )
    )

    imo_by_type_path = os.path.join(
        os.path.expanduser("~"), config["model_data"], "imo_by_type_2030.pkl",
    )
    with open(imo_by_type_path, "rb") as f:
        imo_by_type = pickle.load(f)

    # imos_in_mbd = list(chain.from_iterable([i for i in imo_by_type.values()]))

    # check which imo from ship data base is in AIS (unique) imos
    # model_imos = [i for i in imos_in_mbd if i in unique_imos]

    # len(unique_imos)
    # len(imos_in_mbd)
    # len(model_imos)

    sq = {}
    fs = {}
    for k, v in imo_by_type.items():
        if " FS" in k:
            # get the lenght of imos, if imo of MDB dataset is in AIS unique imo set
            fs[k.replace(" FS", " Tier I")] = len(
                [i for i in v if i in unique_imos]
            )
        else:
            sq[k] = len([i for i in v if i in unique_imos])

    df_2030 = (
        pd.concat([pd.Series(fs, name="FS"), pd.Series(sq, name="SQ")], axis=1)
        .fillna(0)
        .astype("int")
    )
    df_2030.to_csv("tables/old_new_ship_ratio_2030.csv")
    df_2030.to_latex("tables/old_new_ship_ratio_2030.tex")
    # df_2030.sum().sum()
