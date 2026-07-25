import logging
import os


def get_logger(name):

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename="logs/project.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    return logging.getLogger(name)