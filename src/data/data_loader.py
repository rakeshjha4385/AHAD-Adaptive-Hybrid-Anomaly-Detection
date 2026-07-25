"""
Enterprise Dataset Loader

Author : Rakesh Jha

Project : AHAD

"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

from src.config.config import Config


logger = get_logger(__name__)

config = Config()


class DataLoader:

    def __init__(self):

        self.dataset_path = Path(

            config.get("dataset", "path")

        )

        self.filename = config.get(

            "dataset",

            "file_name"

        )

        self.timestamp_column = config.get(

            "dataset",

            "timestamp_column"

        )

    def load(self):

        try:

            file = self.dataset_path / self.filename

            logger.info(f"Loading {file}")

            df = pd.read_csv(file)

            logger.info(f"Dataset Loaded")

            logger.info(f"Shape {df.shape}")

            return df

        except Exception as e:

            logger.exception(e)

            raise

    def parse_timestamp(self, df):

        df[self.timestamp_column] = pd.to_datetime(

            df[self.timestamp_column]

        )

        return df

    def sort_dataset(self, df):

        df = df.sort_values(

            self.timestamp_column

        )

        return df.reset_index(drop=True)

    def remove_duplicates(self, df):

        return df.drop_duplicates()

    def run(self):

        df = self.load()

        df = self.parse_timestamp(df)

        df = self.sort_dataset(df)

        df = self.remove_duplicates(df)

        logger.info("Dataset Ready")

        return df


if __name__ == "__main__":

    loader = DataLoader()

    df = loader.run()

    print(df.head())

    print(df.shape)