import os
import logging
import pickle
import json
import pandas as pd

logger = logging.getLogger(__name__)


def reduce_ais_data(dataset, config, debug=False):
    """ Read raw data files for specified dataset and remove
    obsolete columns, rename and write to disk.
    """

    datapath = os.path.join(
        os.path.expanduser("~"), config["raw_data"], dataset
    )
    processed_path = os.path.join(os.path.expanduser("~"), config["processed"])

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    files = os.listdir(datapath)

    for file in files:
        logger.info("Read and reduce dataset `{}`".format(file))
        filepath = os.path.join(datapath, file)
        filecontent = open(filepath)
        df = pd.read_csv(
            filecontent,
            index_col=0,
            sep=config[dataset]["sep"],
            # nrows=20000,
            usecols=config[dataset]["columns"].keys(),
        )
        df.index = pd.to_datetime(
            df.index, format=config[dataset]["datetimeformat"]
        ).strftime("%Y-%m-%d %H:%M:%S")

        df.index.name = "datetime"

        df.rename(
            columns=config[dataset]["columns"], inplace=True,
        )
        path = os.path.join(
            processed_path, file.replace(".csv", "-reduced.csv")
        )

        logger.info("Write processed file to {}".format(path))

        df.to_csv(path)


def merge_ais_data(config):
    """
    """
    datapath = os.path.join(os.path.expanduser("~"), config["processed"])

    months = ["201412"] + [str(2015) + str(i).zfill(2) for i in range(1, 13)]

    for month in months:
        logging.info("Merge month {}".format(str(month)))
        vessel_file = os.path.join(
            datapath, "vesselfinder_" + month + "-reduced.csv"
        )
        helcom_file = os.path.join(
            datapath, "helcom_" + month + "-reduced.csv"
        )

        vessel_df = pd.read_csv(
            vessel_file,
            index_col=[0],
            parse_dates=True,
        )
        helcom_df = pd.read_csv(
            helcom_file,
            index_col=[0],
            parse_dates=True,
        )

        # first write vesselfinder
        merged_dirpath = os.path.join(
            os.path.expanduser("~"), config["merged"]
        )
        if not os.path.exists(merged_dirpath):
            os.makedirs(merged_dirpath)

        for day in vessel_df.index.day.unique():
            logging.info("Writing merged data for day {}".format(day))
            merged_filepath = os.path.join(
                merged_dirpath, month + str(day).zfill(2) + ".csv"
            )

            vessel_df.loc[vessel_df.index.day == day].to_csv(merged_filepath)

            helcom_day = helcom_df.loc[helcom_df.index.day == day]

            helcom_day[helcom_day["lon"] > vessel_df["lon"].max()].to_csv(
                merged_filepath, mode="a", header=False
            )


def merge_lcpa_models(
    input_path=os.path.join("emission_model", "lcpa-models"),
    lcpa_model_name="lcpa_model.csv",
    config=None,
):
    """ Merges raw lcpa files
    """
    output_path = os.path.join(
        os.path.expanduser("~"), config["model_data"], lcpa_model_name
    )

    logger.info(
        "Merging lcpa models from {0} to one file {1}".format(
            input_path, output_path
        )
    )

    files = os.listdir(input_path)
    lcpa = pd.DataFrame()
    for file in files:
        _df = pd.read_csv(
            os.path.join(input_path, file), sep=";", skiprows=1, index_col=[0]
        )
        lcpa = pd.concat([lcpa, _df])

    lcpa.to_csv(output_path, sep=";")

    return lcpa


def append_additional_emissions_to_lcpa(
    scenario="2015_sq",
    lcpa_model_name="lcpa_model.csv",
    output_dir=None,
    config=None,
):
    """
    """
    lcpa_model_path = os.path.join(
        os.path.expanduser("~"), config["model_data"], lcpa_model_name
    )

    # read the lcpa model to extend be additional pollutants
    df = pd.read_csv(lcpa_model_path, sep=";", index_col=[0, 1, 2])

    # get the maximum speed per shiptype
    # max_speed = df.groupby(level=0).apply(max)["Speed [m/second]"]
    # .to_csv("emission_model/max_speed_per_type.csv")

    def _add_emissions(row, scenario, dataframe):
        """
        """
        energy_factor = (
            row["Energy [J]"] / 3.6e6 / 1e3
        )  # J -> kWh -> kg (s. below)
        fuel_factor = row["Fuel Consumption [kg]"]
        # assign PM value from LCPA tool, will only be replace for
        # "high" scenarios
        pm = row["PM [kg]"]

        # for future scenarios apply different emission factors
        if "FS" in row.name[0] and "low" in scenario:
            bc = 0
            poa = 0
            co = 0
            ash = 0
            nmvoc = 0

            return (bc, ash, poa, co, nmvoc, pm)

        elif "FS" in row.name[0] and "high" in scenario:
            # for "high" future scenario 0 for black carbon ash,
            # rest like SQ
            bc = 0
            ash = 0
            # except pm which is -95% of, replace with 0.05 * Tier II values
            pm = (
                dataframe.loc[
                    row.name[0].replace("FS", "Tier II"),
                    row.name[1],
                    row.name[2],
                ]["PM [kg]"]
                * 0.05
            )

            # auxiliary engine (medium speed diesel)
            if row.name[1] == "Electrical":
                poa = (
                    0.2 * energy_factor * 0.45
                )  # from Schwarzkopf 2016 (0.45 for efficiency losses)
                co = (
                    (6.86 + 5.02) / 2 * fuel_factor / 1e3
                )  # EEA 2021, p.28 avg. value from cruise and hotelling
                nmvoc = (2.6 + 2.04) / 2 * fuel_factor / 1e3

            # main engine
            else:
                poa = 0.2 * energy_factor * 0.45  # from Schwarzkopf 2016

                # types slow speed diesel engine
                if any(
                    i in row.name[0]
                    for i in ["Bulker", "Tanker", "Container", "Car", "MPV"]
                ):
                    co = (
                        2.52 * fuel_factor / 1e3
                    )  # EEA 2021 p.28 for cruise mode
                    nmvoc = (
                        1.33 * fuel_factor / 1e3
                    )  # alternativ factor based on EMEP/EEA for cruise mode

                else:
                    # types for medium speed diesel engine
                    co = 3.47 * fuel_factor / 1e3  # EEA 2021 cruise mode
                    nmvoc = (
                        1.52 * fuel_factor / 1e3
                    )  # alternativ factor based on EMEP/EEA cruise mode

            return (bc, ash, poa, co, nmvoc, pm)

        else:
            if row.name[1] == "Electrical":
                ash = 0.02 * 0.001 * fuel_factor  # Schwarzkopf 2021
                poa = (
                    0.2 * energy_factor * 0.45
                )  # from Schwarzkopf 2016  (0.45 for efficiency losses)
                co = (
                    (6.86 + 5.02) / 2 * fuel_factor / 1e3
                )  # EEA 2021, p.28 avg. value from cruise and hotelling
                nmvoc = (2.6 + 2.04) / 2 * fuel_factor / 1e3  # EEA
                bc = (0.085 + 0.054) / 2 * fuel_factor / 1e3  # EEA
            else:
                poa = (
                    0.2 * energy_factor * 0.45
                )  # from Schwarzkopf 2016  (0.4 for efficiency losses)
                ash = 0.02 * 0.001 * fuel_factor

                if any(
                    i in row.name[0]
                    for i in ["Bulker", "Tanker", "Container", "Car", "MPV"]
                ):
                    nmvoc = (
                        1.33 * fuel_factor / 1e3
                    )  # alternativ factor based on EMEP/EEA
                    co = 2.52 * fuel_factor / 1e3  # EEA 2021 p.21
                    bc = 0.0327 * fuel_factor / 1e3

                else:
                    # types for medium speed diesel engine
                    nmvoc = (
                        1.52 * fuel_factor / 1e3
                    )  # alternativ factor based on EMEP/EEA
                    co = 3.47 * fuel_factor / 1e3
                    bc = 0.0329 * fuel_factor / 1e3

            return (bc, ash, poa, co, nmvoc, pm)

    df[
        ["BC [kg]", "ASH [kg]", "POA [kg]", "CO [kg]", "NMVOC [kg]", "PM [kg]"]
    ] = df.apply(
        _add_emissions,
        axis=1,
        result_type="expand",
        scenario=scenario,
        dataframe=df,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(os.path.join(output_dir, "model_" + scenario + ".csv"), sep=";")


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


def build_imo_lists(config):
    """ Hackisch function to create listes with ships types and corresponding
    list of imo numbers depend on scenario year


    """
    ships = create_ship_dataframe()
    # tc (typeclass mapper)
    tc_mapper = pd.read_csv(
        os.path.join("emission_model", "ship_weightclass_mapper.csv"),
        index_col=0,
    )
    # ships[(ships["BUILT"] != 0) & (ships["STATUS"] == "IN SERVICE/COMMISSION")].min()

    # extract ships per class and write to pickle
    imo_by_type = {}
    for index, row in tc_mapper.iterrows():
        if not " FS" in index:
            imo_by_type[index] = [
                i
                for i in ships[
                    (row["class"] == ships["FSGTYPE"])
                    & (ships["BUILT"] >= float(row["year_lb"]))
                    & (ships["BUILT"] <= float(row["year_ub"]))
                    & (
                        ships[row["weighttype"]]
                        >= float(row["weightclass_lb"])
                    )
                    & (
                        ships[row["weighttype"]]
                        <= float(row["weightclass_ub"])
                    )
                ]["IMO"]
                .to_dict()
                .values()
            ]

    # write the detailled type to the ship DB
    ships["DETAILTYPE"] = None
    for k, v in imo_by_type.items():
        ships.loc[ships[ships["IMO"].isin(v)].index, "DETAILTYPE"] = k

    # create scenarios
    ships_2030 = ships.copy()
    ships_2040 = ships.copy()

    def append_future_suffix(row, year):
        # "<=" means, the ships from 2015 will have been replaced in 2040
        # so 2040 has no old ships (as we do not analyse all ships
        # build after 2015
        if row["BUILT"] <= year:
            # this sucks: naming for ship classes is not very good, so
            # we spilt at Tier and strip some whitspace and add " FS"
            # which should basicalley replace all Tier ship types names with
            # their correct corresponding future scenario name
            # e.g. "Bulker Handy Max Tier I" -> "Bulker Handy Max FS"
            return row["DETAILTYPE"].split("Tier")[0].rstrip() + " FS"
        else:
            return row["DETAILTYPE"]

    ships_2030["DETAILTYPE"] = ships_2030.apply(
        append_future_suffix, axis=1, year=2005
    )
    ships_2040["DETAILTYPE"] = ships_2040.apply(
        append_future_suffix, axis=1, year=2015
    )

    # convert to lists for correct type
    imo_by_type_2030 = {}
    for k, v in (
        ships_2030.set_index("IMO").groupby("DETAILTYPE").groups.items()
    ):
        imo_by_type_2030[k] = v.tolist()

    imo_by_type_2040 = {}
    for k, v in (
        ships_2040.set_index("IMO").groupby("DETAILTYPE").groups.items()
    ):
        imo_by_type_2040[k] = v.tolist()

    # modelpath = os.path.join(
    #     os.path.expanduser("~"),
    #     config["model_data"],
    #     "model_2030_low.csv"
    # )
    # model =pd.read_csv(modelpath, sep=";")
    # modelclasses = model["Type"].unique()
    #
    # for i in imo_by_type_2040:
    #     if i not in modelclasses:
    #         print(i)
    # all(ele in modelclasses for ele in imo_by_type.keys())

    if config is None:
        with open("config.json") as file:
            config = json.load(file)

    imo_path = os.path.join(os.path.expanduser("~"), config["model_data"])
    with open(os.path.join(imo_path, "imo_by_type_2015.pkl"), "wb") as f:
        pickle.dump(imo_by_type, f)
    with open(os.path.join(imo_path, "imo_by_type_2030.pkl"), "wb") as f:
        pickle.dump(imo_by_type_2030, f)
    with open(os.path.join(imo_path, "imo_by_type_2040.pkl"), "wb") as f:
        pickle.dump(imo_by_type_2040, f)


if __name__ == "__main__":

    import json
    from logger import logger

    logging.info("Start data pre processing...")

    with open("config.json") as file:
        config = json.load(file)

    datasets = ["vesselfinder", "helcom"]
    #
    for d in datasets:
        reduce_ais_data(d, config)

    merge_ais_data(config)
