import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Extracting preprocessor object")
            preprocessor_obj = load_object(preprocessor_path)
            logging.info("Define models and hyperparameter grids")

            # Define models with hyperparameter grids
            models = {
                "Random Forest": {
                    "model": RandomForestRegressor(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [5, 10, 15, None],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                },
                "Decision Tree": {
                    "model": DecisionTreeRegressor(),
                    "params": {
                        "max_depth": [5, 10, 15, None],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                },
                "Gradient Boosting": {
                    "model": GradientBoostingRegressor(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2]
                    }
                },
                "Linear Regression": {
                    "model": LinearRegression(),
                    "params": {}
                },
                "K-Neighbors Regressor": {
                    "model": KNeighborsRegressor(),
                    "params": {
                        "n_neighbors": [3, 5, 7, 9, 11],
                        "weights": ['uniform', 'distance'],
                        "algorithm": ['auto', 'ball_tree', 'kd_tree']
                    }
                },
                "XGBRegressor": {
                    "model": XGBRegressor(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0]
                    }
                },
                "CatBoosting Regressor": {
                    "model": CatBoostRegressor(verbose=False),
                    "params": {
                        "iterations": [100, 200, 300],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "depth": [4, 6, 8]
                    }
                },
                "AdaBoost Regressor": {
                    "model": AdaBoostRegressor(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.5, 1.0]
                    }
                }
            }

            # Hyperparameter tuning using GridSearchCV
            model_report = {}
            best_models = {}

            for model_name, model_info in models.items():
                model = model_info["model"]
                params = model_info["params"]

                logging.info(f"Hyperparameter tuning for: {model_name}")

                if params:  # If there are hyperparameters to tune
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=5,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    logging.info(f"Best params for {model_name}: {best_params}")
                else:
                    # No hyperparameters to tune (e.g., Linear Regression)
                    model.fit(X_train, y_train)
                    best_model = model

                # Evaluate on test set
                y_test_pred = best_model.predict(X_test)
                test_score = r2_score(y_test, y_test_pred)
                model_report[model_name] = test_score
                best_models[model_name] = best_model
                logging.info(f"{model_name} - Test R2 Score: {test_score:.4f}")

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = best_models[best_model_name]

            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")
            logging.info(f"Best model R2 score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
