"""
Dataset Detector

This module automatically profiles an input dataset and extracts
metadata required for anomaly detection experiments.

Supported dataset types:
    1. Batch datasets
    2. Time-series datasets
"""

from typing import Optional, Dict, List

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)

from .model import DatasetMetadata


class DatasetDetector:

    TIMESTAMP_COLUMNS = [
        "timestamp",
        "time",
        "date",
        "datetime",
        "event_time",
    ]

    TARGET_COLUMNS = [
        "label",
        "target",
        "class",
        "is_anomaly",
        "anomaly",
    ]

    def detect(self,
               df: pd.DataFrame,
               dataset_name: str = "Dataset") -> DatasetMetadata:
        """
        Detect dataset characteristics.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset

        dataset_name : str
            Name of dataset

        Returns
        -------
        DatasetMetadata
        """

        timestamp_column = self._detect_timestamp_column(df)

        target_column = self._detect_target_column(df)

        dataset_type = (
            "timeseries"
            if timestamp_column
            else "batch"
        )

        numeric_columns, categorical_columns, boolean_columns = \
            self._detect_column_types(df)

        missing_values = int(df.isnull().sum().sum())

        duplicate_rows = int(df.duplicated().sum())

        class_distribution = self._class_distribution(
            df,
            target_column
        )

        feature_count = len(df.columns)
        if timestamp_column:
            feature_count -= 1
        if target_column:
            feature_count -= 1

        return DatasetMetadata(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            rows=len(df),
            columns=len(df.columns),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            boolean_columns=boolean_columns,
            timestamp_column=timestamp_column,
            target_column=target_column,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            class_distribution=class_distribution,
            feature_count=feature_count,
        )

    def _detect_timestamp_column(
        self,
        df: pd.DataFrame
    ) -> Optional[str]:

        for col in df.columns:

            if col.lower() in self.TIMESTAMP_COLUMNS:
                return col

            if is_datetime64_any_dtype(df[col]):
                return col

        return None

    def _detect_target_column(
        self,
        df: pd.DataFrame
    ) -> Optional[str]:

        for col in df.columns:

            if col.lower() in self.TARGET_COLUMNS:
                return col

        return None

    def _detect_column_types(
        self,
        df: pd.DataFrame
    ):

        numeric_columns = []
        categorical_columns = []
        boolean_columns = []

        for col in df.columns:

            if is_bool_dtype(df[col]):
                boolean_columns.append(col)

            elif is_numeric_dtype(df[col]):
                numeric_columns.append(col)

            elif is_datetime64_any_dtype(df[col]):
                continue

            else:
                categorical_columns.append(col)

        return (
            numeric_columns,
            categorical_columns,
            boolean_columns,
        )

    def _class_distribution(
        self,
        df: pd.DataFrame,
        target_column: Optional[str]
    ) -> Optional[Dict]:

        if target_column is None:
            return None

        return (
            df[target_column]
            .value_counts(dropna=False)
            .to_dict()
        )