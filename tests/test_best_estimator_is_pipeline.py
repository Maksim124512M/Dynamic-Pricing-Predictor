import pandas as pd

from src.pipeline import build_pipeline

def test_gridsearch_runs():
    df = pd.read_csv('data/products.csv')

    y = df['revenue_next_7d']
    X = df.drop(columns=['product_id', 'revenue_next_7d'])

    components = build_pipeline()

    y = df['revenue_next_7d']
    X = df.drop(columns=['revenue_next_7d'])

    components['grid'].fit(X, y)

    assert hasattr(components['grid'], 'best_estimator_')