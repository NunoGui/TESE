import pandas as pd
df = pd.read_csv("images.csv")
print(df.columns.tolist())
print(df.head())