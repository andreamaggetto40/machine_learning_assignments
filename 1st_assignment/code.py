# boilerplate code for loading the data
import pandas as pd
from scipy.io import arff

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

# Features amount

features_list = list(df_schizo.columns)

print(features_list)

# Features types

features_types = df_schizo.dtypes

print(features_types)


