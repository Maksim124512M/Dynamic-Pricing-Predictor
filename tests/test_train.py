import pickle

def test_model_training():
    with open('models/revenue_pipeline.pkl') as file:
        pipeline_file = pickle.load(file)

    with open('models/revenue_grid.pkl') as file:
        grid_file = pickle.load(file)

    assert pipeline_file is not None
    assert grid_file is not None
    