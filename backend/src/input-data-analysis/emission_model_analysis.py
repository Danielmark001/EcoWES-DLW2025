import os
import json
from pathlib import PurePath

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("config.json") as file:
    config = json.load(file)
model_data_path = PurePath(os.path.expanduser("~"), config["model_data"])
df = pd.read_csv(
    PurePath(model_data_path, "model_2015_sq.csv"), sep=";", index_col=[0, 1]
)
idx = pd.IndexSlice

shiptypes = set(
    [
        " ".join(i.split(" ")[0:-2])
        for i in df.index.get_level_values(0).unique()
        if not "FS" in i
    ]
)

pollutants = df.columns[8:13].drop("CH4 [kg]")
# pollutants = ["Energy [J]", "NMVOC [kg]"]
# sns.set_style("darkgrid")
for shiptype in shiptypes:
    for pollutant in pollutants:
        if pollutant == "NOx [kg]":
            # pollutant = "Energy [J]"
            engine = "Propulsion"
            # shiptype = "Tanker SuezMax"
            plt.figure()
            ax = sns.lineplot(
                data=df.loc[idx[shiptype + " FS", engine]].reset_index(),
                x="Speed [m/second]",
                y=pollutant,
                label="FS",
            )
            # ax = [pollutant].plot(label="FS")
            sns.lineplot(
                data=df.loc[idx[shiptype + " Tier II", engine]].reset_index(),
                x="Speed [m/second]",
                y=pollutant,
                label="Tier II",
                ax=ax,
            )
            sns.lineplot(
                data=df.loc[idx[shiptype + " Tier I", engine]].reset_index(),
                x="Speed [m/second]",
                y=pollutant,
                label="Tier I",
                ax=ax,
            )
            ax = sns.lineplot(
                data=df.loc[idx[shiptype + " FS", "Electrical"]].reset_index(),
                x="Speed [m/second]",
                y=pollutant,
                label="Aux-FS",
            )
            # ax = [pollutant].plot(label="FS")
            sns.lineplot(
                data=df.loc[
                    idx[shiptype + " Tier II", "Electrical"]
                ].reset_index(),
                x="Speed [m/second]",
                y=pollutant,
                label="Aux-Tier II",
                ax=ax,
            )
            sns.lineplot(
                data=df.loc[
                    idx[shiptype + " Tier I", "Electrical"]
                ].reset_index(),
                x="Speed [m/second]",
                y=pollutant,
                label="Aux-Tier I",
                ax=ax,
            )

            ax.grid()
            ax.set_xlim(0, 13)
            ax.set_title(shiptype)
            ax.legend()
            plt.savefig("figures/{}-model-{}.pdf".format(pollutant, shiptype))

# Plot for speed power curve
# ---------------------------------------------------------------------------
speed_power_path = "nextcloud-znes/KlimaSchiff/fsg/input_lcpa/kW_pro_speed.csv"
df = pd.read_csv(
    os.path.join(os.path.expanduser("~"), speed_power_path),
    skiprows=10,
    index_col=[0, 1],
)
df = df.stack()
df = df.to_frame()
df.reset_index(inplace=True)
df = df.rename(columns={0: "Power in kW", "level_2": "Speed in kn"})
sns.set_style("darkgrid")
ax = sns.relplot(
    data=df,
    x="Speed in kn",
    y="Power in kW",
    hue="Ship Type",
    col="Size Class",
    col_wrap=2,
    kind="line",
    markers=True,
)
plt.savefig(
    "figures/speed_power_curves_by_shiptype.pdf", bbox_inches="tight",
)
