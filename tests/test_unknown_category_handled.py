import pandas as pd

from src.pipeline import build_pipeline

from sklearn.model_selection import train_test_split

def test_unknown_category():
    df = pd.read_csv('data/products.csv')

    y = df['revenue_next_7d']
    X = df.drop(columns=['product_id', 'revenue_next_7d'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    test_df = pd.DataFrame([{
        'category': 'unknown_category',
        'price': 100,
        'reviews': 10,
        'sales_last_7d': 5,
        'revenue_last_7d': 500
    }])

    pipeline = build_pipeline()['pipeline']
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(test_df)
    assert pred.shape == (1,)