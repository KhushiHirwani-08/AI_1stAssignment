import pandas as pd
import numpy as np

# Sample DataFrame
data = {'Student': ['A', 'B', 'C', 'D', 'E'],
        'Score': [85, 90, np.nan, 88, np.nan]}
df = pd.DataFrame(data)

# Mean Imputation
mean_imputed = df.copy()
mean_imputed['Score'].fillna(mean_imputed['Score'].mean(), inplace=True)

# Median Imputation
median_imputed = df.copy()
median_imputed['Score'].fillna(median_imputed['Score'].median(), inplace=True)

# Mode Imputation
mode_imputed = df.copy()
mode_imputed['Score'].fillna(mode_imputed['Score'].mode()[0], inplace=True)

print("Original DataFrame:")
print(df)
print("\nMean Imputed DataFrame:")
print(mean_imputed)
print("\nMedian Imputed DataFrame:")
print(median_imputed)
print("\nMode Imputed DataFrame:")
print(mode_imputed)
