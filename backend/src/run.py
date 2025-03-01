import click
import json
import os

from multiprocessing import Pool
from itertools import repeat

from calculate_routes import calculate_routes
from calculate_emissions import calculate_emissions
from rasterize_points import rasterize_points
from unique_imos import calc_unique_imos

import preprocess as preprocess


@click.group()
@click.option(
    "-p",
    "--preprocess",
    type=bool,
    default=False,
    help="If raw data needs to be preprocessed first",
)
@click.option(
    "--parallel",
    "-l",
    type=int,
    default=0,
    help="Number of processes (int) when using multiprocessing parallel computing",
)
@click.option(
    "-o",
    "--overwrite",
    type=bool,
    default=False,
    help="If True, existing files will be overwritten. Default: False",
)

@click.pass_context
def cli(ctx, preprocess, parallel, overwrite):
    ctx.ensure_object(dict)
    ctx.obj["PREPROCESS"] = preprocess
    ctx.obj["PARALLEL"] = parallel
    ctx.obj["OVERWRITE"] = overwrite

@cli.command()
def reduce_ais():
    with open("config.json") as file:
        config = json.load(file)

    datasets = ["vesselfinder", "helcom"]
    for d in datasets:
        preprocess.reduce_ais_data(d, config)


@cli.command()
def merge_ais():
    with open("config.json") as file:
        config = json.load(file)

    preprocess.merge_ais_data(config)


@cli.command()
def unique_imos():
    calc_unique_imos()


# helper to be used in different commands...
def _build_emission_models(config):

    model_data_path = os.path.join(
        os.path.expanduser("~"), config["model_data"]
    )

    if not os.path.exists(model_data_path):
        os.makedirs(model_data_path)

    preprocess.merge_lcpa_models(config=config)

    scenario_names = [
        "2015_sq",
        "2030_low",
        "2030_high",
        "2040_low",
        "2040_high",
    ]
    if config["scenario"] not in scenario_names:
        raise ValueError(
            "Scenario: {} does not exist.".format(config["scenario"])
        )

    for name in scenario_names:
        preprocess.append_additional_emissions_to_lcpa(
            scenario=name,
            output_dir=os.path.join(
                os.path.expanduser("~"), config["model_data"]
            ),
            config=config,
        )

    preprocess.build_imo_lists(config)


@cli.command()
def build_emission_models():
    with open("config.json") as file:
        config = json.load(file)

    _build_emission_models(config)


@cli.command()
def calc_routes():
    with open("config.json") as file:
        config = json.load(file)
    calculate_routes(config)


@cli.command()
@click.pass_context
def calc_emissions(ctx):
    with open("config.json") as file:
        config = json.load(file)

    # create up-to-date model file
    _build_emission_models(config)
    calculate_emissions(config, columns="all", overwrite=ctx.obj["OVERWRITE"])


@cli.command()
@click.pass_context
def rasterize(ctx):
    with open("config.json") as file:
        config = json.load(file)
    # parallel
    if ctx.obj["PARALLEL"] > 0:
        pool = Pool(processes=ctx.obj["PARALLEL"])
        pool.starmap(
            rasterize_points,
            zip(
                repeat(config),
                [
                    {"SOx [kg]": "SO2"},
                    {"PM [kg]": "PM"},
                    {"NOx [kg]": "NOx"},
                    {"CO2 [kg]": "CO2"},
                    {"BC [kg]": "EC"},
                    {"ASH [kg]": "Ash"},
                    {"POA [kg]": "POA"},
                    {"CO [kg]": "CO"},
                    {"NMVOC [kg]": "NMVOC"},
                ],
            ),
        )

    else:
        rasterize_points(
            config=config,
            emission_types={
                "SOx [kg]": "SO2",
                "PM [kg]": "PM",
                "NOx [kg]": "NOx",
                "CO2 [kg]": "CO2",
                "BC [kg]": "EC",
                "ASH [kg]": "Ash",
                "POA [kg]": "POA",
                "CO [kg]": "CO",
                "NMVOC [kg]": "NMVOC",
            },
        )


@cli.command()
@click.pass_context
def all(ctx):
    """
    """
    with open("config.json") as file:
        config = json.load(file)

    if ctx.obj["PREPROCESS"]:
        datasets = ["vesselfinder", "helcom"]
        for d in datasets:
            reduce_ais(d, config)

        merge_ais(config)

    calculate_routes(config)

    calculate_emissions(config)

    rasterize_points(config=config)


if __name__ == "__main__":
    from logger import logger

    cli(obj={})
