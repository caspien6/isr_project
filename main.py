import pickle
import pandas as pd

datalist = pickle.load(open("dataset.pkl", "rb"))
print(type(datalist))

# Calling DataFrame constructor on list
df = pd.DataFrame(datalist)
print(df)
