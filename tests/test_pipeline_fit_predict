import pandas as pd

from sklearn.model_selection import train_test_split

from src.pipeline import build_pipeline

def test_pipeline_fit_predict(sample_df):
    df = pd.read_csv('data/products.csv')

    y = df['revenue_next_7d']
    X = df.drop(columns=['product_id', 'revenue_next_7d'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    components = build_pipeline()

    y = sample_df['revenue_next_7d']
    X = sample_df.drop(columns=['revenue_next_7d'])

    components['pipeline'].fit(X, y)
    preds = components['pipeline'].predict(X)

    assert len(preds) == len(X)