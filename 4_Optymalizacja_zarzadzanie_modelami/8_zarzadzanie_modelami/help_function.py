from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
import boto3
import os 

def create_model(train_x: pd.DataFrame, 
                 train_y: pd.DataFrame, 
                 model, 
                 num_features: list, 
                 cat_features: list):
    """ Function to create pipeline and fit the model. It contains three steps:
        1. MinMaxScaler for num features
        2. OneHoteEncoder for cat features
        3. Fit given ML model
        params:
            - train_x: pd.DataFrame - data frame with input feautres to the model, 
            - train_y: pd.DataFrame - y values, 
            - model - sklearn model object (not fitted), 
            - num_features: list - list of numercial features in the model, 
            - cat_features: list - list of categorical features in the model
        """
    pass


def save_model_to_s3(s3_client: boto3.client, model,bucket_name: str, path_in_s3:str) -> None:
    """
    Function to save model on S3 bucket.
    params:
        - s3_client - s3 client with access keys
        - model - object with model to load to S3
        - bucket_name - name of the S3 bucket
        - path_in_S3 - path to save the model. It should end with {model_name}.joblib
    """
    pass


def download_model_from_s3(s3_client: boto3.client,bucket_name: str, path_in_s3:str, save_path: str=None) -> None:
    """
    Function to load model from S3 bucket.
    params:
        - s3_client - s3 client with access keys
        - bucket_name - name of the S3 bucket
        - path_in_S3 - path to save the model. It should end with {model_name}.joblib
        - save_path - path on the local machine where the model should be loaded. 
                    if None it will be loaded to the current folder with the same name as in S3.
    """
    pass

def load_model(path: str="model_restaurant_revenue.joblib"):
    """
    Function to load model with *.joblib extension
    Args:
        path (str) - path to the model file. It should ends with file name and extension
    Returns:
        loaded model.
    Exceptions:
        FileNotFoundError - It is raised when there is no model under the path.
    """
    try:
        model = joblib.load(path)
        print('Model zaladowany')
        return model
    except FileNotFoundError:
        print('Model nie został znaleziony')
    

def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict values by the given model and data frame.
    Args:
        model - The scikit-learn fitted model.
        df - df with input variables to predict based on the model
    Returns:
        Given df with new columns pred containing predictions.
    Exceptions:
        Raise an exception when there is no all columns needed to predict in the given df.
    """
    pass
    
    
def predict_user(model) -> str:
    """ 
    Function to predict data given by the user.

    Args: 
        model to make a prediciton
    Retruns:
        String with predicted value.
    Exceptions:
        Raise an exception when prediction method failed.
    """
    pass
    
    
