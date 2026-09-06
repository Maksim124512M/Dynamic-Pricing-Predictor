from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.pipeline import build_pipeline


def test_build_pipeline():
    pipeline = build_pipeline()

    assert type(pipeline) is dict
    assert type(pipeline['pipeline']) is Pipeline
    assert type(pipeline['grid']) is GridSearchCV
