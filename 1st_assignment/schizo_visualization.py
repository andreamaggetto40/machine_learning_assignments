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

# plot of income feature distribution

plt.figure(figsize=(6, 4))
df['income'].value_counts().plot(kind='bar', color=['#4C72B0', '#DD8452'], edgecolor='black')
plt.title('Target Class Distribution: Income')
plt.xlabel('Income Category')
plt.ylabel('Sample Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300)  
plt.close()                                    

# hours.per.week feature distribution plot

plt.figure(figsize=(8, 4))
df['hours.per.week'].dropna().hist(bins=20, edgecolor='black', color='skyblue', grid=False)
plt.title('Distribution of Hours Worked Per Week')
plt.xlabel('Hours per Week')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('hours_distribution.png', dpi=300)
plt.close()

# education vs income relationship plot

plt.figure(figsize=(10, 5))
pd.crosstab(df['education.num'], df['income'], normalize='index').plot(
    kind='bar', stacked=True, color=['#4C72B0', '#DD8452'], edgecolor='black', figsize=(10, 5)
)
plt.title('Proportion of Income Brackets by Years of Education (education.num)')
plt.xlabel('Years of Education (education.num)')
plt.ylabel('Proportion')
plt.legend(title='Income Bracket', labels=['<=50K', '>50K'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('education_vs_income.png', dpi=300)
plt.close()

