import boto3
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import argparse
import os
import warnings
warnings.simplefilter(action='ignore')
import json

print("import your necessary libraries in here") 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


print("enter your project name and enviornment here")
project_name = "aws_workshop"
user_name='mlops'
env = 'dev'

print("loading the stage config data from s3")

def getJsonData(bucket_name,key_name):
    print("[LOG]", bucket_name,'---------')
    print("[LOG]", key_name,'--------------')
      
    s3_client = boto3.client('s3')
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=key_name)
    
    body = csv_obj['Body']
    
    json_string = body.read().decode('utf-8')
    json_content = json.loads(json_string)
    
    return json_content

print("setting the parameters for getJsonData function")
config_bucket = f"dlk-cloud-tier-8-code-ml-{env}"
config_s3_prefix_stage = f'config_files/stage_config/{project_name}/{user_name}/stage_config.json'

print("calling the getJsonData function")
config = getJsonData(config_bucket,config_s3_prefix_stage)

print("json script loaded successfully")


print("enter your own functions in the bellow space")
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
    
    print("encoded successfully")
    return df

def scaling(X):  
    min_max=MinMaxScaler()
    X=pd.DataFrame(min_max.fit_transform(X),columns=X.columns)
    
    return X

print("successfully loaded our own defined functions")


if __name__ == "__main__":

    input_data_path = os.path.join("/opt/ml/processing/input", config['preprocessing']['local_paths']['input1'])
    #input_data_path = os.path.join("/opt/ml/processing/input2", config['preprocessing']['local_paths']['input2'])
    #input_data_path = os.path.join("/opt/ml/processing/input3", config['preprocessing']['local_paths']['input3'])
    #input_data_path = os.path.join("/opt/ml/processing/input4", config['preprocessing']['local_paths']['input4'])


    print("reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)

    print("rename the columns in the dataset")
    df.columns = config['preprocessing']['columns']['selected_columns']
    
    
    #################################### Enter your own script in here ###########################################################################
    
    print("defining the list of categorical variables")
    cat_var = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'Churn']
    
    print("calling our own defined function")
    df = data_manipulation(missing_value(change_format(df)))
    df = cat_encoder(df, cat_var)
    
    print("devide the dataset into X and y")
    X = df.iloc[:, 0:30]
    y = df.iloc[:, -1]
    
    print("scaling the dataset")
    X = scaling(X)
    
    #################################### End of the code #########################################################################################
    
    print("saving the outputs")
    X_output_path = os.path.join("/opt/ml/processing/output1", config['preprocessing']['local_paths']['output1'])
        
    print("saving output to {}".format(X_output_path))
    pd.DataFrame(X).to_csv(X_output_path, header=False, index=False)
    
    y_output_path = os.path.join("/opt/ml/processing/output2", config['preprocessing']['local_paths']['output2'])
    print("saving output to {}".format(y_output_path))
    pd.DataFrame(y).to_csv(y_output_path, header=False, index=False)
    
    """
    y_output_path3 = os.path.join("/opt/ml/processing/output3", config['preprocessing']['local_paths']['output3'])
    print("Saving output to {}".format(y_output_path3))
    pd.DataFrame(y).to_csv(y_output_path3, header=False, index=False)
    """
    
    """
    y_output_path4 = os.path.join("/opt/ml/processing/output4", config['preprocessing']['local_paths']['output4'])
    print("Saving output to {}".format(y_output_path4))
    pd.DataFrame(y).to_csv(y_output_path4, header=False, index=False)
    """
    
    print("successfully completed the preprocessing job")
