import pandas as pd
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt

# income-adult-predictor

# dataset load
dataset_path = "data/income-adult-predictor"
data, meta = arff.loadarff(dataset_path)

# convert to pandas dataframe
df = pd.DataFrame(data)

# ? = missing value. Pre-processing of the dataset to change ? into NaN

df = df.replace("?", np.nan)

# dataset properties

# sample and features amount
sample_amount = df.shape[0]
features_amount = df.shape[1]

# features types
features_types = df.dtypes
print(features_types)

# unique values for key categorical columns
print("income unique values", df["income"].unique())
print("sex unique values:", df["sex"].unique())
print("workclass unique values:", df["workclass"].unique())

# value counts for income feature
print("income value counts:", df["income"].value_counts())

# global numerical min and max across all numerical features
print("Global minimum value across all numerical features:", df.min(numeric_only=True).min())
print("Global maximum value across all numerical features:", df.max(numeric_only=True).max())

# missing values amount 
print("missing values amount: " , df.isna().sum())
print("non-missing values amount : " , df.notna().sum())