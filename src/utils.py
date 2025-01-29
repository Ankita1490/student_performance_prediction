import os
import sys
from typing import Dict
import numpy as np 
import pandas as pd
from src.exception import CustomException
import dill
import configparser
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
       
def catgorise_features(df: pd.DataFrame):
    columns = list(df.columns)
    cat_features, num_features =[],[]
    for col in columns:
        if df[col].dtype == 'O':
            cat_features.append(col)
        else:
            num_features.append(col)
    return cat_features, num_features

def creating_model_parameters(model_parameter:configparser):
    try:
        params ={}
        for key,values in model_parameter.items():
            model_name, param = key.split("_", 1)
            print(model_name)
            if model_name == "decisiontree":
                values = values.split(',')
            else:
                values = [val for val in values.split(',')]
            if model_name not in params:
                params[model_name] ={}
            params[model_name][param] = [float(val) if '.' in val else int(val) if val.isdigit() else val for val in values]
        return params
    except Exception as e:
        raise CustomException(e, sys)
        
        
def evaluate_model(X_train, y_train, X_test, y_test, models, params) -> Dict:

    try:
        report = {}
        for key in models.keys():
            model = models[key]
            param = params[key]
            print(model)
            gs = GridSearchCV(model, param, cv = 3)
            gs.fit(X_train, y_train)

            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            _ = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[key] = test_model_score
        return report            
    except Exception as e:
        CustomException(sys,e)
        
def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(sys,e)
