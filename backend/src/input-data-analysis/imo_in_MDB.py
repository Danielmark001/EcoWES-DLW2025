import os

import pandas as pd
import numpy as np


def create_ship_dataframe():
    type_mapper = pd.read_excel(
        os.path.join("emission_model", "ship_type_fsg_mdb_mapper.xlsx"),
        index_col=0,
    )

    name_mapper = pd.read_excel(
        os.path.join("emission_model", "ship_type_fsg_mdb_mapper.xlsx"),
        sheet_name="FSG_ShipType",
        index_col=0,
    ).to_dict()["fsg_name"]

    ships = pd.read_csv(
        os.path.join(
            os.path.expanduser("~"),
            "klimaschiff",
            "raw_data",
            "MDB-data-complete-area.csv",
        )
    )

    def add_type(row,):
        # import pdb; pdb.set_trace()
        if row["TYPE"] == "0":
            stype = 9
            sname = "Diverse"
        else:
            stype = type_mapper.at[
                row["TYPE"], "fsg_no",
            ]
            sname = name_mapper[stype]

        return (
            stype,
            sname,
        )

    ships[["FSGTYPE", "Class",]] = ships.apply(
        add_type, axis=1, result_type="expand",
    )
    ships = ships.drop(ships.loc[ships["Class"] == "rausnehmen"].index, axis=0)
    ships = ships.drop(ships.loc[ships["BUILT"] > 2015].index, axis=0)

    return ships


ships = create_ship_dataframe()

imos = ships.IMO.unique()

files = os.listdir(
    os.path.join(
        os.path.expanduser("~"), "klimaschiff", "raw_data", "processed"
    )
)

l = []
for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(
            os.path.join(
                os.path.expanduser("~"),
                "klimaschiff",
                "raw_data",
                "processed",
                file,
            )
        )

        month = df.imo.unique()

        len([i for i in np.isin(month, imos) if i == True]) / len(month)
        mask = np.isin(month, imos)
        number, share = len(month), (1 - len(month[mask]) / len(month)) * 100

        l.append((file, number, share))

df = pd.DataFrame(l, columns=["File", "Unique_imos", "Not in"])
df.to_csv("imo_in_MDB_check.csv")
