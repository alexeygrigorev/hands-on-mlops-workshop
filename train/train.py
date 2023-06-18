#!/usr/bin/env python
# coding: utf-8

import os
from datetime import date

import mlflow

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline




def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


def run(train_date: date, val_date: date):
    prefix = 'https://d37ci6vzurychx.cloudfront.net/trip-data'
    train_src = f'{prefix}/green_tripdata_{train_date.year:04d}-{train_date.month:02d}.parquet'
    val_src = f'{prefix}/green_tripdata_{val_date.year:04d}-{val_date.month:02d}.parquet'

    df_train = read_dataframe(train_src)
    df_val = read_dataframe(val_src)

    with mlflow.start_run():
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']

        mlflow.log_params({
            'categorical': categorical,
            'numerical': numerical,
        })

        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values

        model_params = {
            'fit_intercept': True
        }

        mlflow.log_params(model_params)
        
        pipeline = make_pipeline(
            DictVectorizer(),
            LinearRegression(**model_params)
        )
        
        ## train 

        train_dicts = df_train[categorical + numerical].to_dict(orient='records')
        pipeline.fit(train_dicts, y_train)

        ## validate

        val_dicts = df_val[categorical + numerical].to_dict(orient='records')
        y_pred = pipeline.predict(val_dicts)

        ## evaluate

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(rmse)
        
        mlflow.log_metric('rmse', rmse)

        mlflow.sklearn.log_model(pipeline, 'model')

        print(f'run ID: {mlflow.active_run().info.run_id}')


def main():
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000")
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', "nyc-taxi-experiment")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_date = date(year=2022, month=1, day=1)
    val_date = date(year=2022, month=2, day=1)
    run(train_date, val_date)


if __name__ == '__main__':
    main()