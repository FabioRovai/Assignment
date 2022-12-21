import time
from typing import List, Dict

from scipy.stats import wilcoxon
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.base import BaseEstimator


def evaluate_classifiers(classifiers: List[BaseEstimator],
                         X_train, y_train, X_test, y_test) -> List[Dict[str, float]]:
    """
    Evaluate a list of classifiers using various evaluation metrics.

    Parameters:
    classifiers (List[BaseEstimator]): A list of classifiers to be evaluated.
    X_train (pandas.DataFrame): The training data features.
    y_train (pandas.Series): The training data target.
    X_test (pandas.DataFrame): The test data features.
    y_test (pandas.Series): The test data target.

    Returns:
    List[Dict[str, float]]: A list of dictionaries containing the evaluation results for each classifier.
    """
    results = []

    # Train and evaluate each classifier
    for clf in classifiers:
        start = time.time()
        clf.fit(X_train.values, y_train.values)
        end = time.time()
        train_time = end - start

        start = time.time()
        y_pred = clf.predict(X_test.values)
        end = time.time()
        eval_time = end - start

        # Calculate various evaluation scores
        score = clf.score(X_test.values, y_pred)
        recall = recall_score(y_test.values, y_pred)
        precision = precision_score(y_test.values, y_pred)
        f1 = f1_score(y_test.values, y_pred)
        roc_auc = roc_auc_score(y_test.values, y_pred)
        wilcoxon_stat, wilcoxon_pvalue = wilcoxon(y_pred, y_test.values)

        results.append({
            "classifier": clf.__class__.__name__,
            "score": score,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "roc_auc": roc_auc,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_pvalue": wilcoxon_pvalue,
            "train_time": train_time,
            "eval_time": eval_time
        })

    # Sort the results by score in descending order
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
