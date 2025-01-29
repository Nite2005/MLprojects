import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute  import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

from src.mlproject.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        this function is responsible for data tranformation

        
        
        '''
        try:
            numerical_columns = [ "writing_score","reading_score"]
            categorical_columns= [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scalar",StandardScaler())
            ])
            cat_pipeline = Pipeline(steps = [
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scalar",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = [ "writing_score","reading_score"]

           ##divide the dataset to independent and dependent feature

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df = train_df[target_column_name]


            ## divide the dataset to independent and dependent feature
            input_features_test_df =test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and test dataframe")

            input_feature_train_array=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_array=preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(target_features_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_array,np.array(target_features_test_df)
            ]

            logging.info(f"Saved preprocessing object")
            
            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)