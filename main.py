from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.logger import logging

if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        df, path = ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        preprocessor = DataPreprocessing()
        cleaned_df = preprocessor.initiate_data_preprocessing(df)
        logging.info("Data Preprocessing Completed")

        transformation = DataTransformation()
        train_data , test_data ,preprocessor = transformation.initiate_data_transformation(cleaned_df)
        logging.info("Data Transformation Completed")

        
    except Exception as e:
        logging.error(f"Error occurred while testing preprocessing: {e}")
        print(f"❌ Error: {e}")
