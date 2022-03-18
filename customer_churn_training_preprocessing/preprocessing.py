import boto3
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import argparse
import os
import warnings
warnings.simplefilter(action='ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def change_format(df):
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    
    return df

def missing_value(df):
    print("count of missing values: (before treatment)", df.isnull().sum())
    
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    print("count of missing values: (before treatment)", df.isnull().sum())
    print("missing values successfully replaced")
    return df

def data_manipulation(df):
    df = df.drop(['customerID'], axis = 1)
    
    return df

def cat_encoder(df, variable_list):
    dummy = pd.get_dummies(df[variable_list], drop_first = True)
    df = pd.concat([df, dummy], axis=1)
    df.drop(df[cat_var], axis = 1, inplace = True)
    
    print("Encoded successfully")
    return df

def scaling(X):  
    min_max=MinMaxScaler()
    X=pd.DataFrame(min_max.fit_transform(X),columns=X.columns)
    
    return X

if __name__ == "__main__":


    input_data_path = os.path.join("/opt/ml/processing/input", 'telco_cutomer_churn.csv')


    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    
    columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
           'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

    df.columns = columns

    cat_var = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'Churn']
    
    df = data_manipulation(missing_value(change_format(df)))
    df = cat_encoder(df, cat_var)

    X = df.iloc[:, 0:30]
    y = df.iloc[:, -1]
    X = scaling(X)
    
    print("Saving the outputs")
    X_output_path = os.path.join("/opt/ml/processing/output1", "X.csv")   
        
    print("Saving output to {}".format(X_output_path))
    pd.DataFrame(X).to_csv(X_output_path, header=False, index=False)
    
    y_output_path = os.path.join("/opt/ml/processing/output2", "y.csv")   
        
    print("Saving output to {}".format(y_output_path))
    pd.DataFrame(y).to_csv(y_output_path, header=False, index=False)
