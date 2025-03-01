import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

with open("config.json") as file:
    config = json.load(file)

config[
    "intermediate_data"
] = "nextcloud-znes/KlimaSchiff/final_results/analysis"


scenario_path = os.path.join(
    os.path.expanduser("~"), config["intermediate_data"], "carbon_budget.csv",
)

ax = (
    pd.read_csv(scenario_path, index_col=0, sep=",")
    .iloc[0:25]
    .plot(kind="bar", grid=True, color=sns.color_palette("tab20"))
)
ax.set_ylabel("Verbleibendes CO2-Budget in Gg pro Jahr")
ax.set_xlabel("Jahr")
plt.savefig(
    "figures/carbon_budget.pdf", bbox_inches="tight",
)

# ----------------------------------------------------------------------------
scenario_path = os.path.join(
    os.path.expanduser("~"),
    config["intermediate_data"],
    "emission_comparison.csv",
)

df = pd.read_csv(scenario_path, index_col=0, sep=",")

df["Diff MoSES"] = (df["MoSes"] - df["EUF"]) / df["EUF"] * 100
df["Diff STEAM2"] = (df["STEAM2"] - df["EUF"]) / df["EUF"] * 100
df.to_csv("tables/emission_comparion.csv")

# remove BC for better visibilitly
df = df.loc[[i for i in df.index if i != "BC"]]
ax = df[["Diff MoSES"]].plot(
    kind="bar", grid=True, color=sns.color_palette("tab10")
)
ax.set_ylabel("Deviation in %.")
ax.set_xlabel("")
plt.savefig(
    "figures/emission_comparison.pdf", bbox_inches="tight",
)

# -----------------------------------------------------------------------------
scenario_path = os.path.join(
    os.path.expanduser("~"),
    config["intermediate_data"],
    "time_average_speed_per_shiptype.csv",
)

df = pd.read_csv(scenario_path, index_col=0, skiprows=51)
ax = df.T.plot(kind="bar", rot=45, color=sns.color_palette("tab20"), grid=True)
ax.set_xlabel("Speed Intervall in m/s")
ax.set_ylabel("Share of time in %")
plt.savefig(
    "figures/average_speed_per_type.pdf", bbox_inches="tight",
)

pal = sns.color_palette("tab20")
print(pal.as_hex())
