from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post('/predict/', json={
        'product_id': 1,
        'category': 'electronics',
        'price': 100,
        'rating': 4,
        'reviews': 10,
        'discount': 0.1,
        'sales_last_7d': 5,
        'revenue_last_7d': 500
    })

    data = response.json()

    assert response.status_code == 200
    assert 'revenue_next_7d_by_linear_regression' in data
    assert 'revenue_next_7d_by_random_forest' in data
    assert isinstance(data['revenue_next_7d_by_linear_regression'], (int, float))