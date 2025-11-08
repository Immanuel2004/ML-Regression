import os,sys
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing

@dataclass
class DataTransformationConfig:
    transformed_data_dir : str = os.path.join('artifacts', 'transformed_data')
    preprocessor_obj_path : str = os.path.join('artifacts','transformed_data','preprocessor.pkl')
    train_data_path : str = os.path.join('artifacts','transformed_data','train_data.csv')
    test_data_path : str = os.path.join('artifacts','transformed_data','test_data.csv')

class DataTransformation:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_preprocessing = DataPreprocessing()
        self.data_transformation = DataTransformationConfig()
    
    def initiate_data_transformation(self,df:pd.DataFrame,target_column:str=None):
        try:
            logging.info("Data Transformation Started")
            if target_column is None:
                target_column = df.columns[-1]
                logging.info(f"Using Last Column as Target column : {target_column}")

            X = df.drop(columns=[target_column])
            y = df[target_column]

            categorical_cols = X.select_dtypes(include=['object','category']).columns.to_list()
            numeric_cols = X.select_dtypes(include=[np.number]).columns.to_list()
            logging.info(f"Categorical Columns : {categorical_cols}")
            logging.info(f"Numeric Columns {numeric_cols}")

            numeric_transformer = Pipeline(steps=[
                ('scaler',StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num',numeric_transformer,numeric_cols),
                    ('cat',categorical_transformer,categorical_cols)
                ]
            )
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            transformed_feature_names = (
                numeric_cols+
                list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols))
                if categorical_cols else numeric_cols
            )
            train_transformed = pd.DataFrame(X_train_transformed,columns=transformed_feature_names)
            test_transformed = pd.DataFrame(X_test_transformed,columns=transformed_feature_names)

            train_transformed[target_column] = y_train.reset_index(drop=True)
            test_transformed[target_column] = y_test.reset_index(drop=True)

            os.makedirs(self.data_transformation.transformed_data_dir,exist_ok=True)
            train_transformed.to_csv(self.data_transformation.train_data_path, index=False)
            test_transformed.to_csv(self.data_transformation.test_data_path, index=False)

            with open(self.data_transformation.preprocessor_obj_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(f"Transformation completed. Data and preprocessor saved to {self.data_transformation.transformed_data_dir}")

            return train_transformed, test_transformed, preprocessor


        except Exception as e:
            raise CustomException(e,sys)

