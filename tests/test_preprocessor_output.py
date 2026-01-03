import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

def test_preprocessor_output_shape():
    log_transformer = FunctionTransformer(
        np.log1p, feature_names_out='one-to-one'
    )
    
    preprocessor = ColumnTransformer([
        ('num', log_transformer, ['price', 'reviews', 'sales_last_7d', 'revenue_last_7d']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])
    ])

    df = pd.read_csv('data/products.csv')

    X = df.drop(columns=['revenue_next_7d'])
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == len(X)