from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.logger import logging

if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        df, path = ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        preprocessor = DataPreprocessing()
        cleaned_df = preprocessor.initiate_data_preprocessing(df)
        logging.info("Data Preprocessing Completed")

        print("\n✅ Preprocessing completed successfully!")
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"Columns: {list(cleaned_df.columns)}")
        logging.info("Main script executed successfully.")

    except Exception as e:
        logging.error(f"Error occurred while testing preprocessing: {e}")
        print(f"❌ Error: {e}")
