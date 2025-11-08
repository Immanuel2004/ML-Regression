import os ,sys 
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


@dataclass
class ModelTrainerConfig:
    model_dir:str = os.path.join("artifacts","models")
    best_model_path : str = os.path.join("artifacts","models","best_model.pkl")
    model_report_path:str = os.path.join("artifacts","models","model_report.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.model_dir,exist_ok=True)

    def evaluate_model(self,model,X_train,y_train,X_test,y_test):

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test,y_pred)
        mae = np.sqrt(mean_squared_error(y_test,y_pred))
        rmse = mean_absolute_error(y_test,y_pred)

        return {"r2_score": r2, "rmse": rmse, "mae": mae}

    def initiate_model_trainer(self,train_df:pd.DataFrame,test_df:pd.DataFrame,target_column:str,tune_hyperparameters:bool=True):
        try:
            logging.info("Model Training initiated..")

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(verbosity=0, random_state=42),
                "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
            }
            params = {
                "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
                "Ridge": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
                "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]},
                "DecisionTree": {"max_depth": [3, 5, 10, None]},
                "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
                "KNeighbors": {"n_neighbors": [3, 5, 7, 9]},
                "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
                "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.3]},
                "CatBoost": {"depth": [4, 6, 8], "learning_rate": [0.01, 0.1], "iterations": [200, 500]}
            }
            model_report = {}
            best_model = None
            best_r2 = -np.inf
            best_model_name = None

            for name , model in models.items():
                logging.info(f"Training Model : {name}")
                if tune_hyperparameters and name in params:
                    grid = GridSearchCV(model, params[name], scoring='r2', cv=3, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_estimator = grid.best_estimator_
                    logging.info(f"Best params for {name}: {grid.best_params_}")
                else:
                    best_estimator = model

                metrics = self.evaluate_model(best_estimator, X_train, y_train, X_test, y_test)
                model_report[name] = metrics

                if metrics["r2_score"] > best_r2:
                    best_r2 = metrics["r2_score"]
                    best_model = best_estimator
                    best_model_name = name

            report_df = pd.DataFrame(model_report).T
            report_df.to_csv(self.model_trainer_config.model_report_path, index=True)
            logging.info(f"Model report saved at {self.model_trainer_config.model_report_path}")

            joblib.dump(best_model, self.model_trainer_config.best_model_path)
            logging.info(f"Best model '{best_model_name}' saved at {self.model_trainer_config.best_model_path}")

            return {
                "best_model_name": best_model_name,
                "best_r2_score": best_r2,
                "best_model": best_model,
                "model_report": report_df
            }
        except Exception as e:
            raise CustomException(e,sys)
