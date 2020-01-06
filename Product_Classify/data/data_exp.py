import pandas as pd

data = pd.read_csv("train_cut_.csv")

print(len(data))
print(len(list(set(data["TYPE"]))))