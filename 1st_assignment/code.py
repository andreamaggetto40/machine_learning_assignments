# boilerplate code for loading the data
import pandas as pd
from scipy.io import arff

# schizo.arff
data_path = "data/schizo.arff"
data, meta = arff.loadarff(data_path)

df_schizo = pd.DataFrame(data)




