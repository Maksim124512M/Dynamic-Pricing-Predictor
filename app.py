import mlflow
import pandas as pd

from fastapi import FastAPI

from pydantic import BaseModel

from src.inference import predict
from src.config import MLFLOW_MONITORING_CONFIG

mlflow.set_tracking_uri(MLFLOW_MONITORING_CONFIG['tracking_uri'])
mlflow.set_experiment(MLFLOW_MONITORING_CONFIG['experiment_name'])

app = FastAPI(title='Dynamic pricing pipeline')

class InputData(BaseModel):
    product_id: int
    category: str
    price: float
    rating: int
    reviews: int
    discount: float
    sales_last_7d: int
    revenue_last_7d: int

@app.post('/predict/')
async def predict_endpoint(data: InputData) -> dict:
    '''
    Make revenue predictions using trained models.
    '''

    df = pd.DataFrame([data.dict()])

    predictions = predict(df)

    with mlflow.start_run():
        mlflow.log_dict([data.dict()], 'Data from user')

    return {
        'revenue_next_7d_by_linear_regression': float(predictions['pipeline_prediction'][0]),
        'revenue next_7d_by_random_forest': float(predictions['grid_prediction'][0]),
    }