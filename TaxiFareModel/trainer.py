# imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import *
from TaxiFareModel.utils import *
from TaxiFareModel.data import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib


# Constante defined for MLflow only
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [PARIS] [JVESP] Trainer V1"


class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        #extra experiment_name for MLFlow only
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # create distance pipeline
        self.dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())])

        # create time pipeline
        self.time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        # create preprocessing pipeline
        self.preproc_pipe = ColumnTransformer([
            ('distance', self.dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', self.time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Add the model of your choice to the pipeline
        self.pipe = Pipeline([('preproc', self.preproc_pipe),
                              ('linear_model', LinearRegression())])

        # the line below is for mlflow only:
        self.mlflow_log_param('linear model', 'LinearRegression')

    def run(self):
        # train the pipelined model
        self.set_pipeline()
        self.pipe.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)

        # the line below is for mlflow only:
        self.mlflow_log_metric('rmse',rmse)

        return rmse

    ## ADDING ML FLOW
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # Save with JobLib
    def save_model(self):
        joblib.dump(self.pipe, 'model.joblib')



if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df.pop("fare_amount")
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    T = Trainer(X_train,y_train)
    T.run()
    # evaluate
    T.evaluate(X_test, y_test)

    #Savemodel with JobLib
    T.save_model()
