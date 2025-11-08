import os,sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pylab as plt
import seaborn as sns
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion


@dataclass
class DataPreprocessingConfig:
    processed_data_dir : str = os.path.join('artifacts','processed_data')
    correlation_plot_path : str = os.path.join('artifacts','processed_data','correlation.png')
    cleaned_data_path :str = os.path.join('artifacts','processed_data','cleaned_data.csv')

class DataPreprocessing:
    def __init__(self):
        self.preprocessing_config = DataPreprocessingConfig()
        self.data_ingestion = DataIngestion()

    def initiate_data_preprocessing(self,df:pd.DataFrame):
        try:
            logging.info("Started Data Preprocessing and EDA")
            logging.info(f"Initial Dataset Shape {df.shape}")
            logging.info(f"Columns : {list(df.columns)}")

            logging.info("Descriptive Statistical Analysis")
            desc = df.describe(include='all')
            logging.info(f"\n{desc}")

            missing = df.isnull().sum()
            logging.info(f"Missing Values : \n{missing}")

            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                logging.info(f"Found {duplicate_count} duplicate rows and removing them")
                df = df.drop_duplicates()

            for col in df.columns:
                if df[col].dtype in ['int64','float64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            logging.info("Handled All the Missing Values and duplicates")

            constant_cols = [col for col in df.columns if df[col].nunique() == 1]
            if constant_cols:
                logging.info(f"Removing Constant columns : {constant_cols}")
                df.drop(columns=constant_cols,inplace=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.to_list()

            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 - 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                logging.info(f"{col} : {len(outliers)} Outliers Detected")
            logging.info("Removed Outliers")

            os.makedirs(self.preprocessing_config.processed_data_dir,exist_ok=True)

            plt.figure(figsize=(10,6))
            sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(self.preprocessing_config.correlation_plot_path)
            plt.close()
            logging.info("Correlation heatmap saved")

            df.to_csv(self.preprocessing_config.cleaned_data_path,index=False)
            logging.info(f"Cleaned Data saved at : {self.preprocessing_config.cleaned_data_path}")

            logging.info("Data preprocessing and EDA completed successfully.")
            return df

        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise CustomException(e,sys)

        

