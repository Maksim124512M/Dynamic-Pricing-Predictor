<<<<<<< HEAD
<<<<<<< HEAD
=======
import pickle

>>>>>>> 7754be2 (test(pipeline): create test for pipeline building)
from src.pipeline import build_pipeline

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def test_build_pipeline():
<<<<<<< HEAD
    pipeline = build_pipeline()

    assert type(pipeline) is dict
    assert type(pipeline['pipeline']) is Pipeline
    assert type(pipeline['grid']) is GridSearchCV
=======
    result = build_pipeline()

    assert type(result) is dict
    assert type(result['pipeline']) is Pipeline
    assert type(result['grid']) is GridSearchCV
    
>>>>>>> 7754be2 (test(pipeline): create test for pipeline building)
