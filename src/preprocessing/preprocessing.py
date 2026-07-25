"""
AHAD Data Preprocessing Module

Author : Rakesh Jha
"""

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

import pandas as pd

from src.utils.logger import get_logger
from src.config.config import Config

logger = get_logger(__name__)
config = Config()


class DataPreprocessor:

    def __init__(self):

        self.scaler_name = config.get(
            "preprocessing",
            "scaling"
        )

    #######################################################

    def get_numeric_columns(self, df):

        return df.select_dtypes(
            include=["number"]
        ).columns.tolist()

    #######################################################

    def get_scaler(self):

        if self.scaler_name.lower() == "standard":

            return StandardScaler()

        elif self.scaler_name.lower() == "minmax":

            return MinMaxScaler()

        elif self.scaler_name.lower() == "robust":

            return RobustScaler()

        else:

            return None

    #######################################################

    def scale(self, df):

        numeric_cols = self.get_numeric_columns(df)

        if "label" in numeric_cols:

            numeric_cols.remove("label")

        scaler = self.get_scaler()

        if scaler is None:

            logger.info("Scaling Disabled")

            return df

        df_scaled = df.copy()

        df_scaled[numeric_cols] = scaler.fit_transform(

            df_scaled[numeric_cols]

        )

        logger.info(

            f"{self.scaler_name} Scaling Applied"

        )

        return df_scaled

    #######################################################

    def preprocess(self, df):

        return self.scale(df)