import pandas as pd
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATASET LOAD & PRE-PROCESSING
# ==========================================
dataset_path = "data/income-adult-predictor"
data, meta = arff.loadarff(dataset_path)
df = pd.DataFrame(data)

# Decode byte-strings to standard UTF-8 strings and strip padding spaces
for col in df.select_dtypes(include=[object]).columns:
    if isinstance(df[col].iloc[0], bytes):
        df[col] = df[col].str.decode('utf-8').str.strip()

# Replace '?' missing placeholders with NaN
df = df.replace("?", np.nan)


# ==========================================
# 2. DESCRIBING THE DATASET
# ==========================================
print("\n" + "="*40)
print("2. DESCRIBING THE DATASET")
print("="*40)

print("\n--- Dataset Shape & Info ---")
print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
df.info()

print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())

print("\n--- Summary Statistics (Categorical) ---")
print(df.describe(include=["O"]))  # 'O' handles string/categorical object columns


# ==========================================
# 3. ADDITIONAL PROPERTIES & CHECKS
# ==========================================
print("\n" + "="*40)
print("3. FEATURE INSPECTION")
print("="*40)

print("Income unique values:", df["income"].unique())
print("Sex unique values:", df["sex"].unique())
print("Workclass unique values:", df["workclass"].unique())

print("Global minimum value across all numerical features:", df.min(numeric_only=True).min())
print("Global maximum value across all numerical features:", df.max(numeric_only=True).max())


# ==========================================
# 4. DATA ISSUES AUDIT
# ==========================================
print("\n" + "="*40)
print("4. DATA ISSUES AUDIT")
print("="*40)

# 1. Missing Values Check
print("--- Missing Values per Column ---")
missing_counts = df.isnull().sum()
missing_percentage = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame(
    {"Missing Count": missing_counts, "Percentage (%)": missing_percentage}
)
print(missing_df[missing_df["Missing Count"] > 0])

# 2. Duplicate Rows Check
print("\n--- Duplicate Rows ---")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# 3. Target Variable Class (Income) Check
print("\n--- Target Variable Distribution (Income) ---")
print(df["income"].value_counts())
print("\nPercentage Breakdown:")
print((df["income"].value_counts(normalize=True) * 100).round(2))


# ==========================================
# 5. EXPLORATORY DATA ANALYSIS PLOTS
# ==========================================

# Plot 1: Target Feature Distribution (Income)
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

# Plot 2: Hours Per Week Feature Distribution
plt.figure(figsize=(8, 4))
df['hours.per.week'].dropna().hist(bins=20, edgecolor='black', color='skyblue', grid=False)
plt.title('Distribution of Hours Worked Per Week')
plt.xlabel('Hours per Week')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('hours_distribution.png', dpi=300)
plt.close()

# Plot 3: Education vs. Income Relationship
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

print("\nAll description metrics, data audits, and plots completed successfully!")