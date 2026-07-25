"""
Exploratory Data Analysis (EDA)

This module performs automatic exploratory data analysis for
both Time-Series and Batch anomaly detection datasets.

Outputs:
--------
results/
    tables/
        dataset_summary.csv
        missing_values.csv
        feature_statistics.csv

    figures/
        class_distribution.png
        correlation_heatmap.png
        boxplots.png
        time_series.png
        anomaly_plot.png
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.model import DatasetMetadata


class EDA:

    def __init__(self,
                 output_dir: str = "results"):

        self.output_dir = Path(output_dir)

        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    ####################################################################
    # Public Method
    ####################################################################

    def analyze(self,
                df: pd.DataFrame,
                metadata: DatasetMetadata):

        print("=" * 60)
        print("Starting EDA")
        print("=" * 60)

        self.dataset_summary(df, metadata)

        self.missing_value_analysis(df)

        self.feature_statistics(df)

        self.class_distribution(df, metadata)

        if metadata.dataset_type == "timeseries":

            self.time_series_plot(df, metadata)

            self.anomaly_plot(df, metadata)

        else:

            self.correlation_heatmap(df)

            self.boxplots(df)

        print("=" * 60)
        print("EDA Completed Successfully")
        print("=" * 60)

    ####################################################################
    # Dataset Summary
    ####################################################################

    def dataset_summary(self,
                        df,
                        metadata):

        feature_count = metadata.columns

        if metadata.timestamp_column:
            feature_count -= 1

        if metadata.target_column:
            feature_count -= 1

        summary = pd.DataFrame({

            "Attribute": [

                "Dataset Name",
                "Dataset Type",
                "Rows",
                "Columns",
                "Features",
                "Timestamp Column",
                "Target Column",
                "Missing Values",
                "Duplicate Rows"

            ],

            "Value": [

                metadata.dataset_name,
                metadata.dataset_type,
                metadata.rows,
                metadata.columns,
                feature_count,
                metadata.timestamp_column,
                metadata.target_column,
                metadata.missing_values,
                metadata.duplicate_rows

            ]

        })

        summary.to_csv(
            self.tables_dir / "dataset_summary.csv",
            index=False
        )

        print("Dataset Summary Saved")

    ####################################################################
    # Missing Values
    ####################################################################

    def missing_value_analysis(self, df):

        missing = pd.DataFrame({

            "Column": df.columns,

            "Missing Values": df.isnull().sum().values,

            "Percentage":

                (df.isnull().sum() /
                 len(df) * 100).round(2).values

        })

        missing.to_csv(
            self.tables_dir / "missing_values.csv",
            index=False
        )

        print("Missing Value Report Saved")

    ####################################################################
    # Statistics
    ####################################################################

    def feature_statistics(self, df):

        numeric = df.select_dtypes(include=np.number)

        stats = numeric.describe().transpose()

        stats.to_csv(
            self.tables_dir / "feature_statistics.csv"
        )

        print("Feature Statistics Saved")

    ####################################################################
    # Class Distribution
    ####################################################################

    def class_distribution(self,
                           df,
                           metadata):

        if metadata.target_column is None:

            return

        counts = df[
            metadata.target_column
        ].value_counts()

        plt.figure(figsize=(6, 4))

        counts.plot(kind="bar")

        plt.title("Class Distribution")

        plt.xlabel("Class")

        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(
            self.figures_dir /
            "class_distribution.png",
            dpi=300
        )

        plt.close()

        print("Class Distribution Figure Saved")

    ####################################################################
    # Correlation
    ####################################################################

    def correlation_heatmap(self,
                            df):

        numeric = df.select_dtypes(include=np.number)

        corr = numeric.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        image = ax.imshow(
            corr,
            aspect="auto"
        )

        ax.set_xticks(range(len(corr.columns)))

        ax.set_xticklabels(
            corr.columns,
            rotation=90,
            fontsize=8
        )

        ax.set_yticks(range(len(corr.columns)))

        ax.set_yticklabels(
            corr.columns,
            fontsize=8
        )

        plt.colorbar(image)

        plt.tight_layout()

        plt.savefig(

            self.figures_dir /
            "correlation_heatmap.png",

            dpi=300

        )

        plt.close()

        print("Correlation Heatmap Saved")

    ####################################################################
    # Boxplots
    ####################################################################

    def boxplots(self,
                 df):

        numeric = df.select_dtypes(include=np.number)

        if numeric.shape[1] > 10:
            numeric = numeric.iloc[:, :10]

        plt.figure(figsize=(14, 6))

        numeric.boxplot(rot=90)

        plt.tight_layout()

        plt.savefig(

            self.figures_dir /
            "boxplots.png",

            dpi=300

        )

        plt.close()

        print("Boxplots Saved")

    ####################################################################
    # Time Series Plot
    ####################################################################

    def time_series_plot(self,
                         df,
                         metadata):

        timestamp = metadata.timestamp_column

        value = metadata.numeric_columns[0]

        plt.figure(figsize=(15, 5))

        plt.plot(

            df[timestamp],

            df[value],

            linewidth=1

        )

        plt.title("Time Series")

        plt.xlabel("Time")

        plt.ylabel(value)

        plt.tight_layout()

        plt.savefig(

            self.figures_dir /
            "time_series.png",

            dpi=300

        )

        plt.close()

        print("Time Series Figure Saved")

    ####################################################################
    # Anomaly Plot
    ####################################################################

    def anomaly_plot(self,
                     df,
                     metadata):

        if metadata.target_column is None:
            return

        timestamp = metadata.timestamp_column

        value = metadata.numeric_columns[0]

        target = metadata.target_column

        plt.figure(figsize=(15, 5))

        plt.plot(

            df[timestamp],

            df[value],

            linewidth=1

        )

        anomalies = df[df[target] == 1]

        plt.scatter(

            anomalies[timestamp],

            anomalies[value],

            marker="o",

            s=20

        )

        plt.title("Ground Truth Anomalies")

        plt.tight_layout()

        plt.savefig(

            self.figures_dir /
            "anomaly_plot.png",

            dpi=300

        )

        plt.close()

        print("Anomaly Plot Saved")