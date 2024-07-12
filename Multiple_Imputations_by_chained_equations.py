import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Sample DataFrame
data = {'Weight': [70, 80, np.nan, 90, 85],
        'Height': [175, np.nan, 180, 190, 185],
        'Blood Pressure': [120, 130, 125, np.nan, 110]}
df = pd.DataFrame(data)

# MICE Imputer
imputer = IterativeImputer(max_iter=10, random_state=0)
mice_imputed = imputer.fit_transform(df)
mice_imputed_df = pd.DataFrame(mice_imputed, columns=df.columns)

print("Original DataFrame:")
print(df)
print("\nMICE Imputed DataFrame:")
print(mice_imputed_df)
