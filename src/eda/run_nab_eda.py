from pathlib import Path

import pandas as pd

from src.data.dataset_detector import DatasetDetector
from src.eda.eda import EDA


def main():

    # ----------------------------------------------------
    # Dataset Location
    # ----------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "raw" / "timeseries" / "NAB"
    csv_files = sorted(dataset_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under: {dataset_path}"
        )

    # ----------------------------------------------------
    # Load Dataset
    # ----------------------------------------------------

    print("=" * 80)
    print("Loading NAB Dataset Files")
    print("=" * 80)
    print(f"Dataset Root : {dataset_path}")
    print(f"CSV Files Found : {len(csv_files)}")

    for idx, csv_file in enumerate(csv_files, start=1):

        print("\n" + "-" * 80)
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_file.relative_to(dataset_path)}")
        print("-" * 80)

        df = pd.read_csv(csv_file)

        print(f"Dataset Shape : {df.shape}")
        print(df.head())

        detector = DatasetDetector()

        metadata = detector.detect(
            df=df,
            dataset_name=f"NAB/{csv_file.stem}"
        )

        if metadata.timestamp_column:
            df[metadata.timestamp_column] = pd.to_datetime(
                df[metadata.timestamp_column],
                errors="coerce"
            )

        print("\nMetadata")
        print(metadata)

        eda = EDA(output_dir=f"results/nab_eda/{csv_file.stem}")

        eda.analyze(
            df=df,
            metadata=metadata
        )

    print("\nEDA Completed Successfully For All NAB CSV Files")


if __name__ == "__main__":
    main()