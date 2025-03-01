"""
This script cleans the Ship-DB table, as this file is not publicly available
this code can ignore. If the database is available, the `path` needs to be
adapted to the folder where this file is located.
"""
import os

import pandas as pd


path = os.path.join(
    os.path.expanduser("~"), "nextcloud-znes", "KlimaSchiff", "Data", "Ship_DB"
)


df = pd.read_excel(
    os.path.join(path, "Ship-DB-TabD-200220.xlsx"),
    sheet_name="Ship-DB-TabD-200220",
)

df = df.drop(
    columns=[
        "Werftname",
        "Werftort",
        "BauNummer",
        "Ind",
        "Aus",
        "Flagge",
        "Heimathafen",
        "Status",
        "US",
        "Gesellsch",
        "Betreiber",
        "WerftTyp",
        "Kiell",
        "Aufschw",
        "Indienst",
        "Bemerkung",
        "Werft",
        "Bau_Nr",
        "N_Nr",
        "BRT",
        "NRT",
        "TDW",
        "NRZ",
    ]
)

df.drop(
    df[
        (df["Schiffstyp"].isna() == True)
        | (df["Schiffstyp"] == "4")
        | (df["Schiffstyp"] == " ")
    ].index,
    inplace=True,
)

df.to_csv(os.path.join(path, "schiffsdatenbank.csv"))
