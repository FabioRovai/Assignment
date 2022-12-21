

if __name__ == '__main__':

    import json

    # Third party imports
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from imblearn.over_sampling import SMOTE

    # Local module imports
    from src.preprocessing import encode_column
    from engine.best_model import evaluate_classifiers
    from engine.optimisation import get_best_params_for_random_forest,calibrate_random_forest

    # Open and load the data column names from the json file
    with open('json_files/data_cols.json', 'r') as openfile:
        cols = json.load(openfile)

    # Load the training and test data using the column names
    train = pd.read_csv("data/census_income_learn.csv", header=None, names=cols.keys())
    test = pd.read_csv("data/census_income_test.csv", header=None, names=cols.keys())

    # Drop the instance_weight column as it should not be used in the classifiers
    train.drop(columns='instance_weight', inplace=True)
    test.drop(columns='instance_weight', inplace=True)

    # Initialize the label encoder and apply it to the train and test data using the encode_column function
    le = LabelEncoder()
    train = train.apply(encode_column, le=le)
    test = test.apply(encode_column, le=le)

    # Split the data into features (X) and target (y)
    X_train = train.drop(columns="target")
    y_train = train["target"]

    # Use SMOTE to oversample the training data
    rus = SMOTE()
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Split the test data into features (X) and target (y)
    X_test = test.drop(columns="target")
    y_test = test["target"]

    # Initialize the list of classifiers to be evaluated
    classifiers = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier()
    ]

    # Evaluate the classifiers using the evaluate_classifiers function
    results = evaluate_classifiers(classifiers, X_train, y_train, X_test, y_test)

    # Print the results for each classifier

    for result in results:
        print(
            f"{result['classifier']}:"
            f" Score = {result['score']:.3f},"
            f" Recall = {result['recall']:.3f},"
            f" Precision = {result['precision']:.3f}"
            f" F1 = {result['f1']:.3f},"
            f" ROC AUC = {result['roc_auc']:.3f},"
            f" Wilcoxon Stat = {result['wilcoxon_stat']:.3f},"
            f" Wilcoxon p-value = {result['wilcoxon_pvalue']:.3f},"
            f" Train Time = {result['train_time']:.3f}s,"
            f" Eval Time = {result['eval_time']:.3f}s")

    #Quick example for optimisation and calibration
    best_params = get_best_params_for_random_forest(X_train=X_train,
                                                        y_train=y_train,
                                                        random_state=0)

    calibrated_clf = calibrate_random_forest(X_train, y_train, X_test, y_test,
                                             method="isotonic", random_state=0, best_params=best_params)
