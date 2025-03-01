import json
import pickle
import os
import re
import pandas as pd
import zipfile

with open("config.json") as file:
    config = json.load(file)

year = "".join([i for i in config["scenario"] if i.isdigit()])

imo_by_type = os.path.join(
    os.path.expanduser("~"),
    config["model_data"],
    "imo_by_type_" + year + ".pkl",
)

with open(imo_by_type, "rb") as f:
    ships_per_ship_class = pickle.load(f)

scenarios = ["2015_sq", "2030_low", "2030_high", "2040_low", "2040_high"]

for scenario in scenarios:
    scenario_path = os.path.join(
        os.path.expanduser("~"),
        config["intermediate_data"],
        scenario,
        "ship_emissions",
    )
    files = os.listdir(scenario_path)
    files.sort()

    emission_by_shiptype = {}
    for file in files:
        if file.endswith(".zip"):
            df = pd.read_csv(
                os.path.join(scenario_path, file), compression="zip"
            )
            for k, v in ships_per_ship_class.items():
                emission_by_shiptype[(int(re.findall("\d+", file)[0]), k)] = (
                    df.loc[df["imo"].isin(v), df.columns[8:]].sum().values
                )
    emissions_by_type_and_day = pd.DataFrame(
        emission_by_shiptype, index=df.columns[8:]
    ).T

    result_path = os.path.join(
        os.path.expanduser("~"), config["result_data"], scenario
    )

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    emissions_by_type_and_day.to_csv(
        os.path.join(
            result_path,
            "total_emissions_by_type_and_day_" + scenario + ".csv",
        )
    )


# emissions_by_type_and_day["Shipclass"] = [
#     i[1].split(' ')[0] for i in emissions_by_type_and_day.index]
#     emissions_by_type_and_day.reset_index(inplace=True)
#     emissions_by_type_and_day = emissions_by_type_and_day.groupby(["level_0", "Shipclass"]).sum()


# df = pd.read_csv("total_emissions_by_type_and_day.csv", index_col=[0,1], parse_dates=True)
# df.groupby(level=1).sum()
