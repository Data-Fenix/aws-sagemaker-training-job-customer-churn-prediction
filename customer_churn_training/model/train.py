#Import the neccessary libaries in here
import os
import pandas as pd
from xgboost import XGBClassifier,plot_importance
#from imblearn.over_sampling import SMOTE
#from imblearn.combine import SMOTETomek # doctest: +NORMALIZE_WHITESPACE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc,roc_curve,r2_score,confusion_matrix,roc_auc_score,f1_score
from sklearn.model_selection import GridSearchCV
import argparse
import pickle
import boto3
from sklearn.metrics import confusion_matrix , classification_report
import json

project_name = "aws_workshop"
user_name='mlops'
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

env = 'dev'

print("set the parameters")

config_bucket = f"dlk-cloud-tier-8-code-ml-{env}"

config_s3_prefix_stage = f'config_files/stage_config/{project_name}/{user_name}/stage_config.json'
config_s3_prefix_stage1=f'config_files/model_config/{project_name}/{user_name}/model_config.json'

print("calling the getJsonData function")
config = getJsonData(config_bucket,config_s3_prefix_stage)
config1 = getJsonData(config_bucket,config_s3_prefix_stage1)


print("json script loaded successfully")

if __name__ == "__main__":

    training_data_directory = '/opt/ml/input/data/input1/'
    training_data_directory2 = '/opt/ml/input/data/input2/'
    train_features_data = os.path.join(training_data_directory, config['train_model']['local_paths']['input1'])
    train_labels_data = os.path.join(training_data_directory2, config['train_model']['local_paths']['input2'])
    print("Reading input data")
    print("Reading input data from {}".format(train_features_data))
    X = pd.read_csv(train_features_data, header = None)
    
    print("Reading input data from {}".format(train_labels_data))
    y = pd.read_csv(train_labels_data, header = None)
    
    columns = config['train_model']['columns']['selected_columns']

    
    X.columns = columns
    
    column = ['Churn']
    
    y.columns = column
    
    print("Successfully rename the dataset")
    
    print("split the dataset")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)
    
    print("train the model")
    xgb = XGBClassifier()
    parameters = config1['param_grid']
    
    cv = GridSearchCV(xgb, parameters, cv=3)
    
    print("fitting the model")
    cv.fit(X_train, y_train.values.ravel())

    final_model = cv.best_estimator_

    y_pred = final_model.predict(X_test)
    
    print('Model evaluation')

    print(confusion_matrix(y_test,final_model.predict(X_test)))

    print(classification_report(y_test,y_pred))

    roc_auc_score=roc_auc_score(y_test,final_model.predict_proba(X_test)[:, 1])
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    accuracy=accuracy_score(y_test, y_pred)


    print(f"roc_auc_score:{round(roc_auc_score, 3)}")
    print(f"precision:{round(precision, 3)}")
    print(f"recall_score:{round(recall, 3)}")
    print(f"f1_score:{round(f1, 3)}")
    print(f"accuracy_score:{round(accuracy, 3)}")
    
    OUTPUT_DIR = "/opt/ml/model/"
    
    print("Saving model....")
            
    print("Saving model....")
    path = os.path.join(OUTPUT_DIR, config['train_model']['local_paths']['pickle_name'])
    print(f"saving to {path}")
    with open(path, 'wb') as p_file:
        pickle.dump(final_model, p_file)
            
    print('Training Job is completed.')
