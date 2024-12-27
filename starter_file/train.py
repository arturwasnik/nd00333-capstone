from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.workspace import Workspace

def read_data():
       
    df = pd.read_csv('./heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

    x_df = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
    y_df = df[['DEATH_EVENT']]

    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))

    x, y = read_data()

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train.values.ravel())
       
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

if __name__ == '__main__':
    main()