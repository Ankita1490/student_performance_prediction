from cgi import test
import os
import sys 
import configparser
from dataclasses import dataclass
from typing import OrderedDict
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import creating_model_parameters, evaluate_model, save_obj, timer

config = configparser.ConfigParser(dict_type= OrderedDict)
config.read('src\config.ini')
# DEFAULT_PATH = config["DEFAULT_PATH"]["folder"]

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
  
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =ModelTrainerConfig()
    @timer 
    def initiate_model_train(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:,:-1], test_array[:,-1]
            )
            models = {
                "decisiontree" : DecisionTreeRegressor(),
                "randomforest" : RandomForestRegressor(),                
                "adaboostregressor" : AdaBoostRegressor(),
                "gradientboosting": GradientBoostingRegressor(),
                "linearregression":LinearRegression(),
                "kneighborsregressor" :KNeighborsRegressor(),
                #"xgbregressor" :XGBRegressor()
            }
            model_parameters = config["MODEL_PARAMETER"]
            parameters = creating_model_parameters(model_parameters)
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models, parameters)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = {best_model_name:best_model_score}
            if best_model[best_model_name] < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model on test dataset {best_model}")
            save_obj(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = models[best_model_name]
            )
            return best_model
        except Exception as e:
            raise CustomException(sys, e)
        
