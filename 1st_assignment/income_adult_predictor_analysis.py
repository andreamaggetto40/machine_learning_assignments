import pandas as pd
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt

#1.Loading and pre-processing the dataset

dataset_path = "data/income-adult-predictor"
data, meta = arff.loadarff(dataset_path)
df = pd.DataFrame(data)

#here we decode byte-strings to standard UTF-8 strings and strip padding spaces
for col in df.select_dtypes(include=[object]).columns:
    if isinstance(df[col].iloc[0], bytes):
        df[col] = df[col].str.decode('utf-8').str.strip()

#replace '?' missing placeholders with NaN for dataframe
df = df.replace("?", np.nan)


#2.Describing the dataset

print("\n--- Dataset Shape & Info ---")
print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
df.info()

print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())

print("\n--- Summary Statistics (Categorical) ---")
print(df.describe(include=["O"]))  # this 'O' handles string/categorical object columns

#3. Additional properties/checks conducted 

print("Income unique values:", df["income"].unique())
print("Sex unique values:", df["sex"].unique())
print("Workclass unique values:", df["workclass"].unique())

print("Global minimum value across all numerical features:", df.min(numeric_only=True).min())
print("Global maximum value across all numerical features:", df.max(numeric_only=True).max())

#4.Data Issues 

#Missing Values Check
print("--- Missing Values per Column ---")
missing_counts = df.isnull().sum()
missing_percentage = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame(
    {"Missing Count": missing_counts, "Percentage (%)": missing_percentage}
)
print(missing_df[missing_df["Missing Count"] > 0])

#Duplicate Rows Check
print("\n--- Duplicate Rows ---")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

#Target Variable Class (Income) Check
print("\n--- Target Variable Distribution (Income) ---")
print(df["income"].value_counts())
print("\nPercentage Breakdown:")
print((df["income"].value_counts(normalize=True) * 100).round(2))

#5. Exploratory Data Analysis Visuals

#Plot 2: Hours per week distribution (Bin width = 5 hours)
plt.figure(figsize=(10, 5))

# Define bin edges with a step size of 5 hours
min_hours = int(df["hours.per.week"].min())
max_hours = int(df["hours.per.week"].max())
bin_edges = np.arange(min_hours, max_hours + 5, 5)

sns.histplot(
    data=df,
    x="hours.per.week",
    bins=bin_edges,
    kde=True,
    color="darkorange",
    edgecolor="black",
)

plt.xticks(bin_edges[::2])
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.title(
    "Hours Worked per Week Distribution (Bin Width = 5 Hours)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Hours per Week", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

#Plot 4: Age Distribution
plt.figure(figsize=(10, 5))

#Define bin edges starting from min age to max age (please note however that max age is 90)
min_age = int(df["age"].min())
max_age = int(df["age"].max())
bin_edges = np.arange(min_age, max_age + 5, 5)

#histogram with 5-year bin width
sns.histplot(
    data=df,
    x="age",
    bins=bin_edges,
    kde=True,
    color="steelblue",
    edgecolor="black",
)

plt.xticks(bin_edges)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.title("Age Distribution (Bin Width = 5 Years)", fontsize=14, pad=15)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

# Plot 3: Target Class Balance (Income)
plt.figure(figsize=(6, 5))
sns.countplot(x="income", data=df, palette="Set2")
plt.title("Income Class (Target Variable)", fontsize=14)
plt.xlabel("Income Bracket")
plt.ylabel("Count")
plt.show()




print("\nAll description metrics, data audits, and plots completed successfully!")
