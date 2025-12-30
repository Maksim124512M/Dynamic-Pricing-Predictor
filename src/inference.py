import joblib


pipeline = joblib.load('models/revenue_pipeline.pkl')
grid = joblib.load('models/revenue_grid.pkl')

def predict(df):
    '''
    Make predictions using the trained models.
    '''

    pipeline_prediction = pipeline.predict(df)
    grid_prediction = grid.predict(df)

    return {
        'pipeline_prediction': pipeline_prediction,
        'grid_prediction': grid_prediction,
    }