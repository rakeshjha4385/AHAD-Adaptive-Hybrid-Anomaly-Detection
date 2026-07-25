from src.data.data_loader import DataLoader
from src.data.dataset_validator import DatasetValidator

loader = DataLoader()

df = loader.run()

validator = DatasetValidator()

report = validator.run(df)

print(report)