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

if __name__ == "__main__":

    training_data_directory = '/opt/ml/input/data/input1/'
    training_data_directory2 = '/opt/ml/input/data/input2/'
    train_features_data = os.path.join(training_data_directory, "X.csv")
    train_labels_data = os.path.join(training_data_directory2, "y.csv")
    print("Reading input data")
    print("Reading input data from {}".format(train_features_data))
    X = pd.read_csv(train_features_data, header = None)
    
    print("Reading input data from {}".format(train_labels_data))
    y = pd.read_csv(train_labels_data, header = None)
    
    columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                                  'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
                                  'MultipleLines_No phone service', 'MultipleLines_Yes',
                                  'InternetService_Fiber optic', 'InternetService_No',
                                  'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                                  'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                                  'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                                  'TechSupport_No internet service', 'TechSupport_Yes',
                                  'StreamingTV_No internet service', 'StreamingTV_Yes',
                                  'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                                  'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                                  'PaymentMethod_Credit card (automatic)',
                                  'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    
    X.columns = columns
    
    column = ['Churn']
    
    y.columns = column
    
    print("Successfully rename the dataset")
    
    print("split the dataset")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)
    
    print("train the model")
    xgb = XGBClassifier()
    param_grid = {
        'n_estimators': [100, 250, 500],
        'max_depth': [3, 5],
        'learning_rate' : [0.01, 0.05, 0.1],
        'gamma' : [0.0, 0.1, 0.2],
        'min_child_weight' : [1, 3]
    }
    
    cv = GridSearchCV(xgb, param_grid, cv=3)
    
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
    path = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
    print(f"saving to {path}")
    with open(path, 'wb') as p_file:
        pickle.dump(final_model, p_file)
            
    print('Training Job is completed.')
