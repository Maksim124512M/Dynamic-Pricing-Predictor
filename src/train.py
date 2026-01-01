import mlflow
import joblib
import pandas as pd

from src.pipeline import build_pipeline
from src.config import MLFLOW_MONITORING_CONFIG

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

mlflow.set_tracking_uri(MLFLOW_MONITORING_CONFIG['tracking_uri'])
mlflow.set_experiment(MLFLOW_MONITORING_CONFIG['experiment_name'])

def train():
    '''
    Train and save the revenue prediction models.
    '''

    df = pd.read_csv('data/products.csv')

    train_components = build_pipeline()  # Get pipeline and grid search objects

    # Split features and target
    y = df['revenue_next_7d']
    X = df.drop(columns=['product_id', 'revenue_next_7d'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    train_components['pipeline'].fit(X_train, y_train)
    train_components['grid'].fit(X_train, y_train)

    pipeline_y_pred = train_components['pipeline'].predict(X_test)
    grid_y_pred = train_components['grid'].predict(X_test)

    pipeline_rmse = root_mean_squared_error(y_true=y_test, y_pred=pipeline_y_pred)
    pipeline_mae = mean_absolute_error(y_true=y_test, y_pred=pipeline_y_pred)

    grid_rmse = root_mean_squared_error(y_true=y_test, y_pred=grid_y_pred)
    grid_mae = mean_absolute_error(y_true=y_test, y_pred=grid_y_pred)

    # Save the trained models
    joblib.dump(train_components['pipeline'], 'models/revenue_pipeline.pkl')
    joblib.dump(train_components['grid'], 'models/revenue_grid.pkl')

    with mlflow.start_run():
        mlflow.sklearn.log_model(train_components['pipeline'], name='linear_regression_pipeline')
        mlflow.sklearn.log_model(train_components['grid'], name='random_forest_pipeline')

        mlflow.log_params({
            'rf_n_estimators': train_components['grid'].best_estimator_.named_steps['random_forest'].n_estimators,
            'rf_max_depth': train_components['grid'].best_estimator_.named_steps['random_forest'].max_depth,
        })

        mlflow.log_metrics({
            'linear_regression_rmse': pipeline_rmse,
            'linear_regression_mae': pipeline_mae,

            'random_forest_rmse': grid_rmse,
            'random_forest_mae': grid_mae,
        })

if __name__ == '__main__':
    train()