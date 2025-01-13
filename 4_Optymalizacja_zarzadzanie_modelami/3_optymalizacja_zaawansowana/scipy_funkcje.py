from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import  accuracy_score
import pandas as pd

def optimization_function(params, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    """ Function to optimize HistGradientBoostingModel for given train_x (set of input variables) and train_y (output variable).
        It returns accuracy_score based on test_x and test_y
        args:
            params: input params from scipy
            train_x: set of input variables (train)
            train_y: output variable (train)
            test_x: set of input variables (test)
            test_y: output variable (test)
        return: minus accuracy_score (because scipy minimizes function)
    """
    pass    