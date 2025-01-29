import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import catgorise_features, save_obj
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        try:
           cat_features, num_features =catgorise_features(self.train_df) 
           num_features.remove('math score')
           
           num_pipeline =Pipeline(
               steps =[
                   ("imputer", SimpleImputer(strategy= "median")),
                   ("scaler", StandardScaler())
               ]
           )
           categorical_pipleline = Pipeline(
               steps =[
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
               ]
           )
           logging.info("Numerical transformation done")
           logging.info("Categorical transformation done")
           
           preprocessor = ColumnTransformer(
           [
               ("numerical_pipeline", num_pipeline, num_features),
               ("categorical_pipeline", categorical_pipleline, cat_features)
           ]
           )
           return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining Preprocessing object")
            preprocessing_object = self.get_data_transformer_obj()
            target_column_name = "math score"
            
            input_feature_train_df = self.train_df.drop(columns=[target_column_name], axis = 1)
            target_feature_train_df = self.train_df[target_column_name]
            
            input_feature_test_df = self.test_df.drop(columns =[target_column_name], axis =1)
            target_feature_test_df = self.test_df[target_column_name]      
                  
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            input_preprocessing_train_df = preprocessing_object.fit_transform(input_feature_train_df)
            input_preprocessing_test_df = preprocessing_object.transform(input_feature_test_df)
            
            train_arr = np.c_[input_preprocessing_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_preprocessing_test_df,np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object.")
            
            save_obj (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )
            
            return(
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
        except Exception as e:
            raise CustomException(sys, e)
            
            
        

        
        
        
        
    



