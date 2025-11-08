import os 
import sys 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            file_path = input("Enter the filename (with path if not in current directory): ").strip()
            logging.info(f"User provided dataset path: {file_path}")

            df = pd.read_csv(file_path) 
            logging.info("Dataset successfully loaded into a DataFrame.")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            return df, self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise CustomException(e, sys)