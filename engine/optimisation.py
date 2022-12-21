from typing import Dict

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # Do not remove, grays out in pycharm
from sklearn.model_selection import HalvingGridSearchCV


def get_best_params_for_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Dict:
    """
    Get the best parameters for a RandomForestClassifier using HalvingGridSearchCV.

    Parameters:
    X_train (pandas.DataFrame): The training data features.
    y_train (pandas.Series): The training data target.
    random_state (int): The random seed to use for the HalvingGridSearchCV and RandomForestClassifier.

    Returns:
    Dict: A dictionary containing the best parameters for the RandomForestClassifier.
    """
    # Initialize the RandomForestClassifier with a random seed
    clf = RandomForestClassifier(random_state=random_state)

    # Define the parameter grid for the HalvingGridSearchCV
    param_grid = {
        "max_depth": [1, 10]
    }

    # Initialize the HalvingGridSearchCV with the parameter grid, using n_estimators as the resource
    # and a maximum of 10 resources, and setting the random seed
    search = HalvingGridSearchCV(clf, param_grid, resource='n_estimators',
                                 max_resources=10,
                                 random_state=random_state).fit(X_train, y_train)

    # Get the best parameters from the HalvingGridSearchCV
    best_params = search.best_params_

    return best_params






def calibrate_random_forest(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            best_params: Dict,
                            method: str = "isotonic",

                            random_state: int = 0) -> CalibratedClassifierCV:
    """
    Calibrate a RandomForestClassifier using CalibratedClassifierCV.

    Parameters:
    X_train (pandas.DataFrame): The training data features.
    y_train (pandas.Series): The training data target.
    X_test (pandas.DataFrame): The test data features.
    y_test (pandas.Series): The test data target.
    method (str, optional): The calibration method to use. Supported methods are 'sigmoid' and 'isotonic'.
                            The default is 'isotonic'.
    random_state (int, optional): The random seed to use for the RandomForestClassifier. The default is 0.
    best_params (Dict): A dictionary containing the best parameters for the RandomForestClassifier.

    Returns:
    CalibratedClassifierCV: A calibrated RandomForestClassifier.
    """
    # Initialize the RandomForestClassifier with the best parameters and a random seed
    clf = RandomForestClassifier(**best_params, random_state=random_state)

    # Initialize the CalibratedClassifierCV with the RandomForestClassifier and the chosen calibration method
    calibrated_clf = CalibratedClassifierCV(clf, method=method, cv=5).fit(X_train, y_train)

    # Score the calibrated classifier on the test data
    score = calibrated_clf.score(X_test, y_test)
    print(f"Calibrated RandomForestClassifier score on test data: {score:.3f}")

    return calibrated_clf