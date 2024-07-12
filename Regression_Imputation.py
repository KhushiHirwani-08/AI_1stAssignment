import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample DataFrame
data = {'Bedrooms': [3, 4, np.nan, 2, 5],
        'Bathrooms': [2, np.nan, 1, 1, 3],
        'Price': [300000, 400000, 250000, 200000, 500000]}
df = pd.DataFrame(data)

# Define a function for regression imputation
def regression_impute(df, target):
    df_temp = df.copy()
    missing = df_temp[target].isnull()
    model = LinearRegression()
    model.fit(df_temp.loc[~missing, df_temp.columns != target], df_temp.loc[~missing, target])
    df_temp.loc[missing, target] = model.predict(df_temp.loc[missing, df_temp.columns != target])
    return df_temp

# Impute missing values
reg_imputed = regression_impute(df, 'Bedrooms')
reg_imputed = regression_impute(reg_imputed, 'Bathrooms')

print("Original DataFrame:")
print(df)
print("\nRegression Imputed DataFrame:")
print(reg_imputed)
