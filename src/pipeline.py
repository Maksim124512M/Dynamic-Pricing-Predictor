import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import GridSearchCV

def build_pipeline():
    '''
    Build and return the machine learning pipelines.
    '''

    # Define preprocessing steps
    log_transformer = FunctionTransformer(
        np.log1p, feature_names_out='one-to-one'
    )

    # Combine preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer([
        ('num', log_transformer, ['price', 'reviews', 'sales_last_7d', 'revenue_last_7d']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])
    ])

    # Define pipelines for Linear Regression and Random Forest
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('linear_model', LinearRegression())
    ])

    # Define Random Forest pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('random_forest', RandomForestRegressor())
    ])

    grid_params = {
        'random_forest__n_estimators': [100, 300, 500],
        'random_forest__max_depth': [3, 5, 7],
    }

    grid = GridSearchCV(estimator=rf_pipeline, param_grid=grid_params, cv=5, scoring='neg_mean_squared_error')

    return {
        'pipeline': lr_pipeline,
        'grid': grid,
    }