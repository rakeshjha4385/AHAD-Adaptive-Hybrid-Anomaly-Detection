from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import DataPreprocessor

loader = DataLoader()

df = loader.run()

processor = DataPreprocessor()

processed_df = processor.preprocess(df)

print(processed_df.head())