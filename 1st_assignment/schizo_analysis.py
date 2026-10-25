import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np

# Import dataset
data_path = "data/schizo.arff"
data, meta = arff.loadarff(data_path)
schizophrenic_data = pd.DataFrame(data)

# Minor preprocessing to remove leading and trailing "b" and "'"
schizophrenic_data[["target", "sex", "class"]] = schizophrenic_data[["target", "sex", "class"]].astype("string")
schizophrenic_data["target"].str.strip("b'")

# Dataset properties
samples_amount, features_amount = schizophrenic_data.shape
print("Samples amount : " , samples_amount)
print("Features amount : " , features_amount)
features_types = schizophrenic_data.dtypes
print(features_types)
schizophrenic_data["ID"]
print(schizophrenic_data["target"].unique())
print(schizophrenic_data["class"].unique())
print(schizophrenic_data["sex"].unique())
print(schizophrenic_data["ID"].value_counts())
print(schizophrenic_data.drop("ID", axis = 1).min(numeric_only = True).min())
print(schizophrenic_data.drop("ID", axis = 1).max(numeric_only = True).max())
print(schizophrenic_data.isna().sum())
print(schizophrenic_data.notna().sum())

# Target attribute plot
plt.figure()
plt.title("Distribution of class (Target Attribute)")
schizophrenic_data["class"].value_counts().plot.bar(rot = 0)
plt.ylabel("count")

# histogram/plot of selected input attributes 
plt.figure()
plt.title("Distribution of gain_ratio_1 (Bin Width based on Freedman-Diaconis Rule)")
freedman_bins = np.histogram_bin_edges(
    schizophrenic_data["gain_ratio_1"].dropna(), 
    bins = "fd"
)
plt.hist(
    schizophrenic_data["gain_ratio_1"],
    bins = freedman_bins, 
    edgecolor = "black"
)
plt.xlabel("gain_ratio_1")
plt.ylabel("count")

plt.figure()
plt.title("Distribution of target (Input Attribute)")
schizophrenic_data["target"].value_counts().plot.bar(rot = 0)
plt.ylabel("count")

plt.show()
