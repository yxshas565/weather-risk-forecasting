import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

df = pd.read_csv('data/GlobalWeatherRepository.csv')

print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nFirst rows:\n{df.head()}")