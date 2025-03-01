import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

with open("config.json") as file:
    config = json.load(file)


# df_m = df.unstack(level=1).resample("M").sum().unstack().unstack(level=0).swaplevel(0,1)

categories = [
    "Tanker",
    "Bulker",
    "Container",
    "Cruise",
    "Cargo",
    "Ro-Ro",
    "Ro-Pax",
    "MPV",
    "Car Carrier",
    "Diverse",
]


def category(row):
    check = [cat for cat in categories if cat in row]
    if check:
        return check[0]


# aggregated numbers yearly --------------------------------------------
# df_time_agg = df_time.reset_index().groupby("category").sum()
# df_time_agg.to_csv("tables/yearly_emissions_by_shiptype_and_pollutant_SQ.csv")
#
# plot_data = (
#     df_time_agg.drop(["CO2", "NOx", "CO [kg]", "SOx"], axis=1)
#     .stack()
#     .reset_index()
# )
# sns.barplot(x="component", y=0, hue="category", data=plot_data)
# #
#
# ax = df_time_agg[["NOx", "CO [kg]", "SOx"]].divide(1e9).plot(kind="bar")
# ax.set_ylabel("CO2 in Mio ton")
# ax.set_xlabel("Ship type")
# plt.savefig("figures/results/Agg_CO2_emissions_SQ.pdf")

# -----------------------------------------------------------------------------
# scenario comparison
# ---------------------------------------------------------------------------
scenarios = ["2015_sq", "2030_low", "2030_high", "2040_low", "2040_high"]
pollutants = [
    i + " [kg]"
    for i in ["SOx", "NOx", "PM", "CO", "CO2", "ASH", "POA", "NMVOC", "BC"]
]

d_annual = {}
d_daily = {}

for scenario in scenarios:
    scenario_path = os.path.join(
        os.path.expanduser("~"), config["intermediate_data"], scenario,
    )

    df = pd.read_csv(
        os.path.join(
            scenario_path,
            "total_emissions_by_type_and_day_" + scenario + ".csv",
        ),
        parse_dates=True,
    )

    # group by category ("Bulker", "Tanker" etc)
    df["Shipclass"] = df["Unnamed: 1"].apply(lambda x: category(x))

    df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"], format="%Y%m%d")
    # remove 2014 data from results data set
    df = df.set_index("Unnamed: 0")["2015"].reset_index()
    # df.set_index(["Unnamed: 0","Unnamed: 1"], inplace=True)
    df_sums = df.groupby(["Unnamed: 0", "Shipclass"]).sum()

    d_daily[scenario] = df_sums

    df_annual_sums = df_sums.sum(level=1, axis=0).T
    df_annual_sums["Pollutant"] = [
        i.split("-")[1] for i in df_annual_sums.index
    ]
    df_annual_sums["Engine"] = [i.split("-")[0] for i in df_annual_sums.index]
    df_annual_sums["All"] = df_annual_sums.sum(axis=1)
    d_annual[scenario] = df_annual_sums  # .groupby("Pollutant").sum()


# a = d_annual["2015_sq"].groupby("Pollutant").sum().loc[pollutants].T
# a = a.div(1e6) # kg -> Gg
# a.columns = [i.strip(" [kg]") for i in a.columns]
# a.to_latex(
#     "tables/annual_emissions_Gg_per_type_{}.tex".format(scenario),
#     label="tab:annual_emissions_Gg_per_type_{}".format(scenario),
#     caption="Annual emissions for each shiptype in Gg in the scenario {}.".format(
#         scenario
#     ),
#     float_format="{:0.2f}".format,
# )

# ----------------------------------------------------------------------------
# annual sums per per ship type
# ----------------------------------------------------------------------------

for scenario in d_daily:
    _df = d_daily[scenario].T
    _df["Pollutant"] = [row.split("-")[1] for row in df_annual_sums.index]
    _df = _df.groupby("Pollutant").sum().loc[pollutants].T
    tuples = _df.index.map(lambda x: (x[1], pd.to_datetime(x[0])))
    _df.index = pd.MultiIndex.from_tuples(tuples, names=["class", "date"])
    _df = _df.sum(level=0).div(1e6)
    _df.loc["All"] = _df.sum()
    _df.columns = [i.replace(" [kg]", " (Gg/year)") for i in _df.columns]
    _df.sort_index(inplace=True)
    _df.to_csv("tables/annual_emissions_Gg_per_type_{}.csv".format(scenario))
    _df.to_latex(
        "tables/annual_emissions_Gg_per_type_{}.tex".format(scenario),
        label="tab:annual_emissions_Gg_per_type_{}".format(scenario),
        caption="Annual emissions for each shiptype in Gg in the scenario: {}.".format(
            scenario
        ),
        float_format="{:0.0f}".format,
    )

# ----------------------------------------------------------------------------
# timeseries plot average daily emissions
# ----------------------------------------------------------------------------
scenario = "2015_sq"
pollutant = "CO2 [kg]"
_df = d_daily[scenario].T
_df["Pollutant"] = [row.split("-")[1] for row in df_annual_sums.index]
_df = _df.groupby("Pollutant").sum().loc[pollutants].T
tuples = _df.index.map(lambda x: (x[1], pd.to_datetime(x[0])))
_df.index = pd.MultiIndex.from_tuples(tuples, names=["class", "date"])
_df = _df.T.unstack()
# _df = _df.div(1e6)
for shipclass in _df.index.get_level_values(0).unique():
    data = _df.unstack()
    positions = [
        p
        for p in data.index.get_level_values(1).unique()
        if p.hour == 0 and p.is_month_start and p.month in range(1, 13, 1)
    ]
    labels = [l.strftime("%m") for l in positions]
    ax = (
        data.loc[shipclass, pollutant]
        .resample("1M")
        .mean()
        .div(1e6)  # kg -> Gg
        .plot(style="-o", grid=True, label=shipclass)
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Emissions in Gg")
    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel("Month")

    lgd = ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    # ax.set_title(
    #     "Average daily {}-Emissions in {}".format(
    #         pollutant.strip(" [kg]"), scenario
    #     )
    # )
plt.savefig(
    "figures/results/average_daily_{}_emissions_{}.pdf".format(
        pollutant.split(" ")[0], scenario
    ),
    bbox_extra_artists=(lgd,),
    bbox_inches="tight",
)
data[pollutant].unstack(level=0).resample("1M").mean().div(1e6).to_csv(
    "tables/average_daily_{}_emissions_{}.csv".format(
        pollutant.split(" ")[0], scenario
    )
)
data[pollutant].unstack(level=0).resample("1M").mean().div(1e6).to_latex(
    "tables/average_daily_{}_emissions_{}.tex".format(
        pollutant.split(" ")[0], scenario
    ),
    label="tab:average_daily_{}_emissions_{}".format(
        pollutant.split(" ")[0], scenario
    ),
    caption="Average daily {} emissions in scenario {} in Gg.".format(
        pollutant.split(" ")[0], scenario
    ),
    float_format="{:0.2f}".format,
)

# ---------------------------------------------------------------------------
# Share of aux of total emissions -
# ---------------------------------------------------------------------------

for scenario in scenarios:
    temp = d_annual[scenario].groupby(["Engine", "Pollutant"]).sum()
    temp = temp.loc["Electrical"].div(
        temp.loc["Propulsion"] + temp.loc["Electrical"]
    )
    temp = temp.loc[pollutants + ["Energy [J]"]]
    temp.dropna(how="all", inplace=True)
    temp.index = [i.strip("[kg]") for i in temp.index]
    temp.index = [i.strip("[J]") for i in temp.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(temp, annot=True)
    ax.set_title(
        "Share of auxiliary engine emissions of total emissions {}.".format(
            scenario
        )
    )
    plt.savefig(
        "figures/results/heatmap_aux_share_of_total_emissions_{}.pdf".format(
            scenario
        ),
        bbox_inches="tight",
    )
    temp.to_csv(
        "tables/heatmap_aux_share_of_total_emissions_{}.csv".format(scenario)
    )
    temp.to_latex(
        "tables/heatmap_aux_share_of_total_emissions_{}.tex".format(scenario),
        label="tab:heatmap_aux_share_of_total_emissions_{}".format(scenario),
        caption="Heatmap data for emission reduction by pollutant in scenario {}.".format(
            scenario
        ),
        float_format="{:0.2f}".format,
    )


# ---------------------------------------------------------------------------
# heat map and annual data
# ----------------------------------------------------------------------------
pd.DataFrame({k: v.sum(axis=1) for k, v in d_annual.items()}).to_csv(
    "tables/annual_emissions_kg_all_scenarios_by_engine.csv"
)
pd.DataFrame(
    {k: v.groupby("Pollutant").sum().sum(axis=1) for k, v in d_annual.items()}
).to_csv("tables/annual_emissions_kg_all_scenarios_total.csv")

heatmap = {}
for scenario in d_annual:
    d_annual[scenario].to_csv(
        "tables/annual_emission_kg_by_type_{}.csv".format(scenario)
    )
    ref = d_annual["2015_sq"].groupby("Pollutant").sum()  # .loc[pollutants]
    hm = d_annual[scenario].groupby("Pollutant").sum().div(ref.values).sub(1)
    hm.index = [i.replace(" [kg]", "") for i in hm.index]
    hm.index = [i.replace(" [J]", "") for i in hm.index]
    hm.drop(["CH4", "GWP", "NPV [EUR]", "CH4 (Well to tank)"], inplace=True)
    WTT = [
        "CO2 (Well to tank)",
        "NOx (Well to tank)",
        "PM (Well to tank)",
        "Energy (Well to tank)",
        "SOx (Well to tank)",
    ]
    # hm = hm.round(2)
    heatmap[scenario] = hm

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        hm.drop(WTT, axis=0),
        cmap="YlGnBu_r",
        cbar_kws={"label": "Deviation to SQ"},
        annot=True,
        ax=ax,
        fmt=".1%",
    )

    plt.savefig(
        "figures/results/heatmap_emission_reduction_{}.pdf".format(scenario),
        bbox_inches="tight",
    )
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        hm.loc[WTT],
        cmap="YlGnBu_r",
        cbar_kws={"label": "Deviation to SQ"},
        annot=True,
        ax=ax,
        fmt=".1%",
    )
    plt.savefig(
        "figures/results/heatmap_wtt_emission_reduction_{}.pdf".format(
            scenario
        ),
        bbox_inches="tight",
    )
    plt.clf()
    heatmap[scenario].to_csv(
        "tables/heatmap_emission_reduction_{}.csv".format(scenario)
    )
    heatmap[scenario].to_latex(
        "tables/heatmap_emission_reduction_{}.tex".format(scenario),
        label="tab:heatmap_emission_reduction_{}".format(scenario),
        caption="Heatmap with share of auxiliary of total emissions {}".format(
            scenario
        ),
        float_format="{:0.2f}".format,
    )


# co2 balance --------------------------
co2 = {}
for scenario in d_annual:
    co2[scenario] = (
        d_annual[scenario]
        .groupby("Pollutant")
        .sum()
        .loc["CO2 (Well to tank) [kg]"]
        + d_annual[scenario].groupby("Pollutant").sum().loc["CO2 [kg]"]
    )
total_co2_reduction = (
    pd.DataFrame(co2).div(co2["2015_sq"].values, axis=0).sub(1).mul(-100)
)
# fig, ax = plt.subplots(figsize=(6, 6))

ax = total_co2_reduction.iloc[:, [2, 4]].plot(
    kind="barh", color=["lightskyblue", "purple"], grid=True
)
lgd = ax.legend(
    title="Scenario year",
    labels=["2030", "2040"],
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
ax.set_xlim(0, 100)
ax.set_xlabel("Reduction compared to 2015_sq in %")
plt.savefig(
    "figures/results/total_CO2_reduction_compared_to_2015_sq.pdf",
    bbox_inches="tight",
)
total_co2_reduction.to_csv(
    "tables/total_CO2_reduction_compared_to_2015_sq.csv"
)
total_co2_reduction.to_latex(
    "tables/total_CO2_reduction_compared_to_2015_sq.tex",
    label="tab:total_CO2_reduction_compared_to_2015_sq",
    caption="Total CO\textsubscript{2} reduction in 2030 and 2040 compared to 2015 in \%",
    float_format="{:0.2f}".format,
)


# costs ----------------------------------------------------------------------
costs = pd.read_excel(
    os.path.join(os.path.expanduser("~"), cost_path),
    sheet_name="summary",
    index_col=0,
)
costs.rename(index={"PM2.5": "PM", "NOX": "NOx", "SO2": "SOx"}, inplace=True)
costs_d = {}
for scenario in scenarios:
    temp = d_annual[scenario].groupby(["Engine", "Pollutant"]).sum()
    temp = temp.sum(level=1)
    temp.index = [i.replace(" [kg]", "") for i in temp.index]
    temp.index = [i.replace(" [J]", "") for i in temp.index]
    temp = temp.loc[["NMVOC", "SOx", "PM", "NOx", "CO2"], "All"]
    temp = temp.div(1000)  # kg -> t

    costs_d[(scenario, "NOx")] = dict(temp.loc["NOx"] * costs.loc["NOx"])
    costs_d[scenario, "PM"] = dict(temp.loc["PM"] * costs.loc["PM"])
    costs_d[scenario, "SOx"] = dict(temp.loc["SOx"] * costs.loc["SOx"])
    costs_d[scenario, "NMVOC"] = dict(temp.loc["NMVOC"] * costs.loc["NMVOC"])
    costs_d[scenario, "CO2 (UBA low)"] = dict(
        temp.loc["CO2"] * costs.loc["CO2 (UBA low)"]
    )
    costs_d[scenario, "CO2 (UBA high)"] = dict(
        temp.loc["CO2"] * costs.loc["CO2 (UBA high)"]
    )
    costs_d[scenario, "CO2 (ETS low)"] = dict(
        temp.loc["CO2"] * costs.loc["CO2 (ETS low)"]
    )
    costs_d[scenario, "CO2 (ETS high)"] = dict(
        temp.loc["CO2"] * costs.loc["CO2 (ETS high)"]
    )
df = pd.DataFrame(costs_d).T
df.to_csv("tables/costs.csv")
df = df.div(1e9)
df.sort_index(inplace=True)
df_select = df.loc[
    ["2015_sq", "2030_high", "2040_high"],
    ["CO2 (UBA low)", "CO2 (UBA high)", "CO2 (ETS low)", "CO2 (ETS high)"],
    :,
]

data = df_select.stack().reset_index()
data.columns = ["Scenario", "Pollutant", "Level", "Costs in Billion Euro"]
g = sns.catplot(
    x="Pollutant",
    y="Costs in Billion Euro",
    hue="Scenario",
    col="Level",
    data=data,
    kind="bar",
    height=4,
    aspect=0.7,
    dodge=True,
    palette=sns.color_palette("tab20"),
)
axes = g.axes.flatten()
g.set_xticklabels(rotation=45)
axes[0].set_title("Low")
axes[1].set_title("Medium")
axes[2].set_title("High")
plt.savefig(
    "figures/CO2_costs.pdf", bbox_inches="tight",
)

df_select_2 = df.loc[
    ["2015_sq", "2030_high", "2040_high"], ["NOx", "PM", "SOx"], :
]

data = df_select_2.stack().reset_index()
data.columns = ["Scenario", "Pollutant", "Level", "Costs in Billion Euro"]
g = sns.catplot(
    x="Pollutant",
    y="Costs in Billion Euro",
    hue="Scenario",
    col="Level",
    data=data,
    kind="bar",
    height=4,
    aspect=0.7,
    dodge=True,
    palette=sns.color_palette("tab20"),
)
axes = g.axes.flatten()
g.set_xticklabels(rotation=45)
axes[0].set_title("Low")
axes[1].set_title("Medium")
axes[2].set_title("High")
plt.savefig(
    "figures/pollutant_costs.pdf", bbox_inches="tight",
)
