import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocess import create_ship_dataframe

tc_mapper = pd.read_csv(
    os.path.join("emission_model", "ship_weightclass_mapper.csv"), index_col=0
)
ships = create_ship_dataframe()

# remove ships built after 2015
ships = ships[(ships["BUILT"] <= 2015) & (ships["BUILT"] > 1900)]


mapping = {
    "Diverse": "Diverse",
    "Cargo": "MPV",
    "RoRo Passenger": "Ro-Pax",
    "Cruise Liner": "Cruise",
    "Tanker": "Tanker",
    "Bulker": "Bulker",
    "RoRo Ship": "Ro-Ro",
    "Container": "Container",
    "RoRo Car": "Car Carrier",
}

ships["NEWCLASS"] = ships["Class"].map(mapping)
# tc_mapper.loc[[i for i in tc_mapper.index if "Tier II" in i]]

# Analysis --------------------------------------------------------------------
gt_classes = {
    "Ro-Ro": [(0, 25e3), (25e3, float("+inf"))],
    "Ro-Pax": [(0, 25e3), (25e3, float("+inf"))],
    "Cruise": [(0, 25e3), (25e3, float("+inf"))],
    "Diverse": [(0, 2e3), (2e3, float("+inf"))],
    "Container": [
        (0, 17.5e3),
        (17.5e3, 55e3),
        (55e3, 145e3),
        (145e3, float("+inf")),
    ],
    "Car Carrier": [(0, 40e3), (40e3, float("+inf"))],
}

dwt_classes = {
    "Tanker": [(0, 35e3), (35e3, 45e3), (45e3, 120e3), (120e3, float("+inf"))],
    "Bulker": [(0, 35e3), (35e3, 45e3), (45e3, 120e3), (120e3, float("+inf"))],
    "MPV": [(0, 120e3), (120e3, float("+inf"))],
}
# a = ships[ships.Class=="Diverse"].groupby("TYPENAME").count()
# (a / a.IMO.sum() * 100).IMO.to_csv("share_diverse.csv")
# shiptype and weigth plot --------------------------------------------------
ships[["NEWCLASS", "GT", "BUILT"]]
cutter = [
    (0, 2000),
    (5000, 10e3),
    (10e3, 30e3),
    (30e3, 60e3),
    (60e3, 145e3),
    (145e3, 10000e3),
]

ships["GT Class"] = pd.cut(
    ships["GT"], pd.IntervalIndex.from_tuples(cutter), precision=0,
)

df = ships.groupby(["NEWCLASS", "GT Class"])["IMO"].count().reset_index()
df = df.rename(
    columns={
        "IMO": "Count",
        "NEWCLASS": "Shipclass",
        "GT Class": "Gross tonnage",
    }
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Gross tonnage", y="Count", hue="Shipclass", data=df)
ax.set_ylabel("Number of Ships")
# ax.set_xlabel("Gross tonnage", size=14)
ax.set_xticklabels(
    ["<" + str((int(i[1]))) for i in cutter[0:-1]] + [">145000"]
)
ax.legend(title="Type")

plt.savefig("figures/shiptype_by_gt.pdf")
df.to_latex(
    "tables/shiptype_by_gt.tex",
    caption="All ships by class and gross tonnage weigth as of 2015",
    label="fig:shiptype_by_gt",
)
# Weight distribution---------------------------------------------------------
# for class
sns.histplot(
    data=ships[ships["NEWCLASS"].isin(["Tanker"])],
    x="DWT",
    hue="NEWCLASS",
    multiple="stack",
    bins=100,
)
["GT"].hist(bins=100)


# Age structure --------------------------------------------------------------
# cutter = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 100)]
# cutter_years = [(2015-i[1], 2015-i[0]) for i in cutter]
ships = ships[(ships["BUILT"] <= 2015) & (ships["BUILT"] > 1950)]

plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data=ships[["NEWCLASS", "BUILT"]],
    x="BUILT",
    hue="NEWCLASS",
    multiple="stack",
)
leg = ax.get_legend()
leg.set_title("Type")
ax.set_ylabel("Number of Ships", size=14)
ax.set_xlabel("Built year", size=14)
plt.savefig("figures/age_structure_by_shiptype.pdf")

# GWT mean
import json

with open("config.json") as file:
    config = json.load(file)

unique_imos = pd.read_csv(
    os.path.join(
        os.path.expanduser("~"), config["model_data"], "unique_imos.csv",
    ),
    index_col=0,
)

# remove ships from mdb which are not in vesselfinde or helcom data set 2015
ships = ships.loc[ships.IMO.isin(unique_imos["0"])]

gt_d = {}
for k, cutter in gt_classes.items():
    ships_by_class = ships[ships["NEWCLASS"] == k]
    vals = pd.cut(
        ships_by_class["GT"],
        pd.IntervalIndex.from_tuples(cutter),
        precision=0,
    )
    gt_d[k] = ships_by_class.groupby(vals).count()["IMO"]
df_gt = pd.DataFrame(gt_d)
df_gt["Weighttype"] = "GT"
df_gt.set_index("Weighttype", append=True, drop=True, inplace=True)
dwt_d = {}
for k, cutter in dwt_classes.items():
    ships_by_class = ships[ships["NEWCLASS"] == k]
    vals = pd.cut(
        ships_by_class["DWT"],
        pd.IntervalIndex.from_tuples(cutter),
        precision=0,
    )
    dwt_d[k] = ships_by_class.groupby(vals).count()["IMO"]
df_dwt = pd.DataFrame(dwt_d)
df_dwt["Weighttype"] = "DWT"
df_dwt.set_index("Weighttype", append=True, drop=True, inplace=True)
combined = pd.concat([df_dwt, df_gt])
combined.to_csv("tables/number_of_ships_per_weightclass.csv")
combined.fillna(0).astype("int").to_latex(
    "tables/number_of_ships_per_weightclass.tex",
    caption="Number of ships per shiptype and weightclass.",
    label="tab:number_of_ships_per_weightclass",
)
