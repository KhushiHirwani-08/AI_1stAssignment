import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Sample DataFrame
data = {'Age': [25, np.nan, 35, 40, np.nan],
        'Income': [50000, 60000, np.nan, 80000, 70000]}
df = pd.DataFrame(data)

# KNN Imputer
imputer = KNNImputer(n_neighbors=2)
knn_imputed = imputer.fit_transform(df)
knn_imputed_df = pd.DataFrame(knn_imputed, columns=df.columns)

print("Original DataFrame:")
print(df)
print("\nKNN Imputed DataFrame:")
print(knn_imputed_df)
