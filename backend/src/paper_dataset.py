import pickle
import os
from itertools import chain

import pandas as pd
from pathlib import PosixPath

nice_type_names = pd.read_csv(
    "emission_model/short_long_name_mapper.csv", index_col=0
).to_dict()["short_name"]
# create path for results
publication_path = os.path.join(
    os.path.expanduser("~"), "klimaschiff", "publication"
)

if not os.path.exists(publication_path):
    os.makedirs(publication_path)

# create new unique ids and get imo_by_type dict
p = PosixPath("~/klimaschiff/model_data/imo_by_type_2015.pkl")
with open(p.expanduser(), "rb") as f:
    imos = pickle.load(f)
imo_list = list(chain.from_iterable(imos.values()))
anonym_id_mapper = dict(zip(imo_list, range(len(imo_list))))

p = PosixPath("~/klimaschiff/model_data/model_2015_sq.csv")
model = pd.read_csv(p.expanduser(), sep=";")
model.drop(
    [
        "Input (delta) [%]",
        "CH4 (Well to tank) [kg]",
        "GWP [kg]",
        "CH4 [kg]",
        "NPV [EUR]",
    ],
    axis=1,
    inplace=True,
)


def map_nicenames(row):
    return nice_type_names[row["Type"]]


model["Type"] = model.apply(map_nicenames, axis=1)
model.rename(columns={"Component": "Engine"}, inplace=True)
#model.set_index(["Type", "Component", "Speed [m/second]"], inplace=True)
# reorder ...by hand :-(
model = model[
    [
        "Type",
        "Engine",
        "Speed [m/second]",
        "Energy (Well to tank) [J]",
        "CO2 (Well to tank) [kg]",
        "SOx (Well to tank) [kg]",
        "NOx (Well to tank) [kg]",
        "PM (Well to tank) [kg]",
        "Energy [J]",
        "Fuel Consumption [kg]",
        "CO2 [kg]",
        "SOx [kg]",
        "NOx [kg]",
        "PM [kg]",
        "BC [kg]",
        "ASH [kg]",
        "POA [kg]",
        "CO [kg]",
        "NMVOC [kg]",
    ]
]
p = PosixPath("~/klimaschiff/publication/model.csv")
#model.columns  = [c.capitalize() for c in model.columns]
model.to_csv(p.expanduser(), index=False)


# loop over
p = PosixPath("~/klimaschiff/intermediate_data/2015_sq/ship_emissions")
p = p.expanduser()

droplist = [
    "Propulsion-NPV [EUR]",
    "Electrical-NPV [EUR]",
    "Propulsion-GWP [kg]",
    "Electrical-GWP [kg]",
    "Propulsion-CH4 [kg]",
    "Electrical-CH4 [kg]",
    "Propulsion-CH4 (Well to tank) [kg]",
    "Electrical-CH4 (Well to tank) [kg]",
    "tdiff",
    "dist",
]


def map_imo_type(row):
    return [nice_type_names[k] for k, v in imos.items() if int(row["imo"]) in list(v)][0]

def map_imo_anonym(row):
    return anonym_id_mapper[row["imo"]]

l = [
    "Energy (Well to tank) [J]",
    "CO2 (Well to tank) [kg]",
    "SOx (Well to tank) [kg]",
    "NOx (Well to tank) [kg]",
    "PM (Well to tank) [kg]",
    "Energy [J]",
    "Fuel Consumption [kg]",
    "CO2 [kg]",
    "SOx [kg]",
    "NOx [kg]",
    "PM [kg]",
    "BC [kg]",
    "ASH [kg]",
    "POA [kg]",
    "CO [kg]",
    "NMVOC [kg]",
]

for file in p.iterdir():
    df = pd.read_csv(file)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df.drop(droplist, axis=1, inplace=True)
    df.insert(0, "Type", df.apply(map_imo_type, axis=1))
    df.insert(0, "UniqueID", df.apply(map_imo_anonym, axis=1))
    df.drop("imo", axis=1, inplace=True)
    # reorder columns
    new_col_order = [j for j in chain(*[["Propulsion-" + i, "Electrical-"+i] for i in l])]
    df = df[list(df.columns[0:6]) + new_col_order]
    df.columns = [c[0].upper() + c[1:] for c in df.columns]
    compression_opts = dict(method="zip", archive_name=file.stem + ".csv")
    df.to_csv(
        os.path.join(publication_path, file.stem + ".zip"),
        index=False,
        compression=compression_opts,
    )
