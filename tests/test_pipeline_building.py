import pickle

from src.pipeline import build_pipeline

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def test_build_pipeline():
    result = build_pipeline()

    assert type(result) is dict
    assert type(result['pipeline']) is Pipeline
    assert type(result['grid']) is GridSearchCV
    