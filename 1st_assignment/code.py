# boilerplate code for loading the data
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np

# schizo.arff
data_path = "data/schizo.arff"
data, meta = arff.loadarff(data_path)

df_schizo = pd.DataFrame(data)

'''Ideas for plotting / visualization:
- Basic information about the dataset (number of samples, number of features, feature names, data types)
- Target attribute distribution
- Histogram of the distribution of the target variable (schizo vs. non-schizo)
- Box plots of the features to visualize their distributions and identify potential outliers
- Pair plots to visualize relationships between features and the target variable
- Numerical Input Attributes (Value Ranges & Distributions)
- Categorical Attributes (Types & Cardinality)
- Feature Correlations & Relationships
'''

# Shape of dataset

samples_amount, features_amount = df_schizo.shape

print("Samples amount : " , samples_amount)
print("Features amount : " , features_amount)

# Features types
features_types = df_schizo.dtypes
print(features_types)
df_schizo['ID']
print(df_schizo['target'].unique())
print(df_schizo['class'].unique())
print(df_schizo['sex'].unique())
print(df_schizo['ID'].unique())
print(df_schizo.drop("ID", axis = 1).min(numeric_only = True).min())
print(df_schizo.drop("ID", axis = 1).max(numeric_only = True).max())
# target plot
df_schizo['class'].value_counts().plot.bar(rot = 0)

# histogram/plot of selected input variables 
fig, axes = plt.subplots(ncols = 2)
freedman_bins = np.histogram_bin_edges(df_schizo['gain_ratio_1'].dropna(), bins = "fd")
df_schizo.hist(column = "gain_ratio_1", bins = freedman_bins, grid = False, ax = axes[0])
df_schizo['target'].value_counts().plot.bar(rot = 0, ax = axes[1])
plt.show()
