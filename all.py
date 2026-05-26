# ================================
# EDA LAB EXPERIMENTS 1 TO 11
# Simple Concepts + Important Code
# ================================

# ---------- IMPORT LIBRARIES ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

# ---------- LOAD DATASET ----------
df = pd.read_csv("your_dataset.csv")

# =========================================
# EXP 1 : LOADING AND EXPLORING DATASET
# =========================================

print(df.head())          # first 5 rows
print(df.tail())          # last 5 rows
print(df.info())          # structure
print(df.describe())      # statistics

# =========================================
# EXP 2 : DATA TYPES & BASIC STATISTICS
# =========================================

print(df.dtypes)

num_cols = df.select_dtypes(include=np.number)
cat_cols = df.select_dtypes(include='object')

print("Numerical Columns:")
print(num_cols.columns)

print("Categorical Columns:")
print(cat_cols.columns)

print("Mean:\n", num_cols.mean())
print("Median:\n", num_cols.median())
print("Min:\n", num_cols.min())
print("Max:\n", num_cols.max())
print("Std:\n", num_cols.std())

# =========================================
# EXP 3 : COMPLETE EDA WORKFLOW
# =========================================

print(df.shape)
print(df.isnull().sum())

sns.histplot(df[num_cols.columns[0]])
plt.show()

sns.boxplot(x=df[num_cols.columns[0]])
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# =========================================
# EXP 4 : HANDLING MISSING VALUES
# =========================================

print(df.isnull().sum())

# deletion
df_drop = df.dropna()

# mean
for col in num_cols.columns:
    df[col] = df[col].fillna(df[col].mean())

# median
for col in num_cols.columns:
    df[col] = df[col].fillna(df[col].median())

# mode
for col in cat_cols.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df.isnull().sum())

# =========================================
# EXP 5 : OUTLIER DETECTION & TREATMENT
# =========================================

# IQR METHOD
Q1 = df[num_cols.columns[0]].quantile(0.25)
Q3 = df[num_cols.columns[0]].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df[num_cols.columns[0]] < lower) |
    (df[num_cols.columns[0]] > upper)
]

print(outliers)

# Z-SCORE METHOD
z = np.abs(zscore(df[num_cols.columns[0]]))
print(df[z > 3])

# Visualization
sns.boxplot(x=df[num_cols.columns[0]])
plt.show()

# =========================================
# EXP 6 : DATA TRANSFORMATION
# =========================================

# log transformation
df["log_data"] = np.log1p(df[num_cols.columns[0]])

# sqrt transformation
df["sqrt_data"] = np.sqrt(df[num_cols.columns[0]])

# scaling
scaler = MinMaxScaler()
df["scaled"] = scaler.fit_transform(df[[num_cols.columns[0]]])

# Visualization
plt.hist(df[num_cols.columns[0]])
plt.title("Before")
plt.show()

plt.hist(df["log_data"])
plt.title("After Log")
plt.show()

# =========================================
# EXP 7 : NORMALIZATION & STANDARDIZATION
# =========================================

# Min-Max Scaling
minmax = MinMaxScaler()
df["MinMax"] = minmax.fit_transform(df[[num_cols.columns[0]]])

# Z-score Standardization
standard = StandardScaler()
df["Zscore"] = standard.fit_transform(df[[num_cols.columns[0]]])

print(df[["MinMax", "Zscore"]].head())

# =========================================
# EXP 8 : FEATURE ENGINEERING
# =========================================

# age groups
df["Age_Group"] = pd.cut(
    df[num_cols.columns[0]],
    bins=[0,20,40,60,100],
    labels=["Young","Adult","Senior","Old"]
)

# salary category example
# df["Salary_Category"] = pd.cut(df["Salary"],
# bins=[0,30000,60000,100000],
# labels=["Low","Medium","High"])

print(df.head())

# =========================================
# EXP 9 : DATA INTEGRATION & REDUCTION
# =========================================

# second dataset
df2 = pd.read_csv("second_dataset.csv")

# merge
merged = pd.merge(df, df2, on="id")

print(merged.head())

# correlation
corr = merged.corr(numeric_only=True)

print(corr)

# remove highly correlated columns
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

drop_cols = [column for column in upper.columns if any(upper[column] > 0.9)]

reduced_df = merged.drop(columns=drop_cols)

print(reduced_df.head())

# =========================================
# EXP 10 : VISUALIZATION
# =========================================

# BAR CHART
df[cat_cols.columns[0]].value_counts().plot(kind='bar')
plt.show()

# PIE CHART
df[cat_cols.columns[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()

# HISTOGRAM
plt.hist(df[num_cols.columns[0]])
plt.show()

# DENSITY PLOT
sns.kdeplot(df[num_cols.columns[0]], fill=True)
plt.show()

# =========================================
# EXP 11 : RELATIONSHIP & MULTIVARIATE ANALYSIS
# =========================================

# Scatter Plot
sns.scatterplot(
    x=df[num_cols.columns[0]],
    y=df[num_cols.columns[1]]
)
plt.show()

# Pair Plot
sns.pairplot(df[num_cols.columns[:4]])
plt.show()

# Correlation Matrix
corr = df.corr(numeric_only=True)

# Heatmap
sns.heatmap(corr, annot=True)
plt.show()

# =========================================
# END
# =========================================
