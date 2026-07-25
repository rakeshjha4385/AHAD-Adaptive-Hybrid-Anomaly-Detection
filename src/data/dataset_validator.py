"""
Enterprise Dataset Validator

Author : Rakesh Jha

Project : AHAD

"""

import pandas as pd
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.config.config import Config

logger = get_logger(__name__)
config = Config()


class DatasetValidator:

    def __init__(self):

        self.timestamp_column = config.get(
            "dataset",
            "timestamp_column"
        )

        self.target_column = config.get(
            "dataset",
            "target_column"
        )

    ##########################################################

    def check_shape(self, df):

        return {

            "rows": df.shape[0],

            "columns": df.shape[1]

        }

    ##########################################################

    def check_missing(self, df):

        return df.isnull().sum().to_dict()

    ##########################################################

    def check_duplicates(self, df):

        return int(df.duplicated().sum())

    ##########################################################

    def check_timestamp(self, df):

        try:

            pd.to_datetime(df[self.timestamp_column])

            return "Valid"

        except Exception:

            return "Invalid"

    ##########################################################

    def check_sorted(self, df):

        return bool(
            df[self.timestamp_column].is_monotonic_increasing
        )

    ##########################################################

    def check_datatypes(self, df):

        return {

            column: str(dtype)

            for column, dtype in df.dtypes.items()

        }

    ##########################################################

    def class_distribution(self, df):

        if self.target_column not in df.columns:

            return {}

        return (

            df[self.target_column]

            .value_counts()

            .to_dict()

        )

    ##########################################################

    def validate(self, df):

        report = {

            "shape": self.check_shape(df),

            "missing_values": self.check_missing(df),

            "duplicates": self.check_duplicates(df),

            "timestamp_validation": self.check_timestamp(df),

            "sorted_by_time": self.check_sorted(df),

            "data_types": self.check_datatypes(df),

            "class_distribution": self.class_distribution(df)

        }

        return report

    ##########################################################

    def save_report(

            self,

            report,

            output_path="outputs"

    ):

        Path(output_path).mkdir(

            exist_ok=True

        )

        report_file = Path(

            output_path

        ) / "dataset_validation_report.json"

        with open(

                report_file,

                "w"

        ) as file:

            json.dump(

                report,

                file,

                indent=4

            )

        logger.info(

            f"Validation report saved : {report_file}"

        )

    ##########################################################

    def run(self, df):

        report = self.validate(df)

        self.save_report(report)

        return report