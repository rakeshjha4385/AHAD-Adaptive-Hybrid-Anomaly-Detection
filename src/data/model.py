from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class DatasetMetadata:
    dataset_name: str

    dataset_type: str

    rows: int
    columns: int

    numeric_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]

    timestamp_column: Optional[str]
    target_column: Optional[str]

    missing_values: int
    duplicate_rows: int

    class_distribution: Optional[Dict]
    
    feature_count: int