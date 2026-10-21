# boilerplate code for loading the data
import pandas as pd
from scipy.io import arff

# schizo.arff
data_path = "data/schizo.arff"
data, meta = arff.loadarff(data_path)

df_schizo = pd.DataFrame(data)

'''Ideas for plotting / visualization:
- Target attribute distribution
- Histogram of the distribution of the target variable (schizo vs. non-schizo)
- Box plots of the features to visualize their distributions and identify potential outliers
- Pair plots to visualize relationships between features and the target variable
- Numerical Input Attributes (Value Ranges & Distributions)
- Categorical Attributes (Types & Cardinality)
- Feature Correlations & Relationships
'''



