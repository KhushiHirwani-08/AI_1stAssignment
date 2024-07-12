import pandas as pd
import numpy as np

# Sample Time-Series DataFrame
dates = pd.date_range('20230101', periods=10)
data = {'Temperature': [30, np.nan, 32, 31, np.nan, 35, np.nan, 33, 34, np.nan]}
df = pd.DataFrame(data, index=dates)

# Forward Fill Imputation
ffill_imputed = df.copy()
ffill_imputed['Temperature'].fillna(method='ffill', inplace=True)

# Backward Fill Imputation
bfill_imputed = df.copy()
bfill_imputed['Temperature'].fillna(method='bfill', inplace=True)

# Interpolation Imputation
interpolated = df.copy()
interpolated['Temperature'].interpolate(method='linear', inplace=True)

print("Original DataFrame:")
print(df)
print("\nForward Fill Imputed DataFrame:")
print(ffill_imputed)
print("\nBackward Fill Imputed DataFrame:")
print(bfill_imputed)
print("\nInterpolated DataFrame:")
print(interpolated)

