# import functools
import os
import logging
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def interpolate_emissions(
    routes,
    emission_types,
    ship_classes,
    engine_types,
    ships_per_ship_class,
    model_data,
    resample,
):
    """ Simple way of the fitting solution in "emissions_by_type_and_class()"
    to get emissions
    """
    emissions = {}
    for ship_class in ship_classes:

        # TODO: Check if ship_class exists if not log warning!
        ship_imo_numbers = ships_per_ship_class.get(ship_class, [])
        # if ship_imo_numbers == []:
        #     logging.info(
        #         "No ship type {} in imo-number dict. This may be correct.".format(
        #             ship_class
        #         )
        #     )

        x = routes.loc[routes["imo"].isin(ship_imo_numbers)]

        if not x.empty:
            for emission_type in emission_types:
                for engine in engine_types:
                    x.loc[:, (engine + "-" + emission_type)] = np.interp(
                        x["speed_calc"],
                        model_data.loc[(ship_class, engine)][
                            "Speed [m/second]"
                        ].values,
                        model_data.loc[(ship_class, engine)][
                            emission_type
                        ].values,
                    ) / (
                        60 / int(resample)
                    )  # convert from hourly model values to "resample" minutes
            emissions[ship_class] = x
        else:
            emissions[ship_class] = None
    return emissions


def read_routes(filepath):
    """
    """
    routes = pd.read_csv(filepath)  # usecols=["imo", "speed_calc"]
    nans = routes.imo.isna().sum()
    if nans > 0:
        logging.warning(
            "`{}` NANs in ships routes removed from file: `{}`".format(
                nans, filepath
            )
        )
    routes = routes.dropna(how="any")
    routes["imo"] = routes.imo.astype("int")
    return routes


def calculate_emissions(config, columns=["CO2 [kg]"], overwrite=False):
    """
    """

    # with open("config.json") as file:
    #     config = json.load(file)
    datapath = os.path.join(
        os.path.expanduser("~"), config["intermediate_data"], "ship_routes"
    )

    outputpath = os.path.join(
        os.path.expanduser("~"),
        config["intermediate_data"],
        config["scenario"],
        "ship_emissions",
    )

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    filepaths = [os.path.join(datapath, i) for i in os.listdir(datapath)]

    # get scenario-specific dict with mapper for imo-number to model
    imo_by_type = os.path.join(
        os.path.expanduser("~"),
        config["model_data"],
        "imo_by_type_"
        + "".join(i for i in config["scenario"] if i.isdigit())
        + ".pkl",
    )
    with open(imo_by_type, "rb") as f:
        ships_per_ship_class = pickle.load(f)

    model_name = os.path.join(
        os.path.expanduser("~"),
        config["model_data"],
        "model_" + config["scenario"] + ".csv",
    )

    model_data = pd.read_csv(model_name, sep=";", index_col=[0, 1])

    # zipped_ship_routes = zipfile.ZipFile(
    #     os.path.join(outputpath, "ship_routes.zip"),
    #     mode="w",
    #     compression=zipfile.ZIP_DEFLATED,
    # )

    if columns == "all":
        emission_types = model_data.columns[2:]
    else:
        emission_types = columns

    for filepath in filepaths:
        routes = read_routes(filepath)

        outputfile = os.path.join(
            outputpath,
            os.path.basename(filepath)
            .replace("ship_routes", "ship_emissions")
            .replace("csv", "zip"),
        )

        if Path(outputfile).is_file() and overwrite is False:
            logging.warning(
                "Skip writing {} because already exists and overwrite is set to False.".format(
                    outputfile
                )
            )
        else:
            logging.info("Writing emissions to file {}.".format(outputfile))
            emissions = interpolate_emissions(
                routes,
                ship_classes=[
                    i
                    for i in model_data.index.get_level_values(0).unique()
                    # if not " FS" in i
                ],
                engine_types=[
                    i for i in model_data.index.get_level_values(1).unique()
                ],  # ["Electrical", "Propulsion"]
                emission_types=emission_types,
                ships_per_ship_class=ships_per_ship_class,
                model_data=model_data,
                resample=config["resample"],
            )
            model_data.index.get_level_values(0).unique()

            df_emissions = pd.concat(emissions.values()).sort_index()

            filename = os.path.basename(filepath).replace(
                "ship_routes", "ship_emissions"
            )

            compression_options = dict(method="zip", archive_name=filename)
            df_emissions.to_csv(outputfile, compression=compression_options)

        # # Write to zip file
        # zipped_ship_routes.write(
        #     outputfile, arcname=os.path.basename(outputfile)
        # )
        # os.remove(outputfile)
    # zipped_ship_routes.close()
