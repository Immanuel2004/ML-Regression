from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
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

        trainer = ModelTrainer()
        result = trainer.initiate_model_trainer(
            train_df=train_data,
            test_df=test_data,
            target_column=train_data.columns[-1],   
            tune_hyperparameters=True                
        )
        best_model_name = result["best_model_name"]
        best_r2 = result["best_r2_score"]
        logging.info(f"Best Model: {best_model_name} | R² Score: {best_r2:.4f}")
        print(f"\nBest Model: {best_model_name}")
    except Exception as e:
        logging.error(f"Error occurred while testing preprocessing: {e}")
        print(f"Error: {e}")
