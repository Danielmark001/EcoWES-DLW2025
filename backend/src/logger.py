import os
import logging

logging.basicConfig(
    filename=os.path.join(
        os.path.expanduser("~"), "klimaschiff", "klimaschiff.log"
    ),
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()
logging.info("Start logging...")
logger.setLevel(logging.INFO)


# logging.info("Start calculating ship routes.")
# calculate_routes(dataset, config)
