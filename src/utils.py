import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException

def upload_file()->pd.DataFrame:
    try:
        df = input("Enter the Filename : ")
        logging.info("User have uploaded the dataset ")
        return df
    except Exception as e:
        raise CustomException(e,sys)


