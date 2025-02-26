import numpy as np
import os
import json
import time
import subprocess


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


from helper import consts
from helper import utils

def tune_hyperparams_dt(X, y):
    space = {
        "max_depth": [3,5,10,15,20],
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    clf = DecisionTreeClassifier()
    search = GridSearchCV(clf, space, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    result = search.fit(X, y)
    return result

def train_python_dt(X, y):
    """
    Trains a decision tree classifier using nested cross validation.
    :param X: DataFrame containing input features in training set
    :param y: Series containing target variables in training set
    :return: dict containing fitted estimator, best parameters, and model performance attributes
    """
    print(utils.CYAN + "Training Python DT" + utils.RESET)
    print(f"Features: {X.columns}")
    start_ts = time.time()
    outer_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for _fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        tuned = tune_hyperparams_dt(X_train, y_train)
        best_clf = tuned.best_estimator_
        y_train_pred = best_clf.predict(X_train)  

        pred_start = time.perf_counter_ns()
        y_pred = best_clf.predict(X_test)
        pred_end = time.perf_counter_ns()
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        fold_res = {
            "train_accuracy": train_accuracy,
            "est_accuracy": tuned.best_score_,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "batch_pred_time": (pred_end - pred_start) / len(y_pred),
            "params": tuned.best_params_,
        }
        outer_results.append(fold_res)
    accuracy = np.mean([fold_res["accuracy"] for fold_res in outer_results])
    precision = np.mean([fold_res["precision"] for fold_res in outer_results])
    recall = np.mean([fold_res["recall"] for fold_res in outer_results])
    f1 = np.mean([fold_res["f1"] for fold_res in outer_results])
    pred_time = np.mean([fold_res["batch_pred_time"] for fold_res in outer_results])
    
    # train final model (apply inner CV to entire dataset)
    tuned = tune_hyperparams_dt(X, y)
    clf = tuned.best_estimator_
    params = tuned.best_params_
    
    print(f"training time: {time.time() - start_ts}s")
    res = {
        "clf": clf, 
        "params": params, 
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
        "batch_pred_time": pred_time,
    }
    return res

def test_python_dt(clf, X_test, y_test, num_trials=1000):
    """
    Run sequential prediction on a test set and measure the prediction times.
    :param clf: Trained classifier
    :param X_test: DataFrame containing test features
    :param y_test: DataFrame containing test labels 
    :param num_trials: int number of trials to measure, len(X_test) if None
    :return: dict containing Python model performance attributes
    """
    print(utils.CYAN + "Testing Python DT" + utils.RESET)
    num_trials = min(num_trials, len(X_test))
    y_hat = []
    y_time = []
    indexes = np.random.choice(len(X_test), num_trials, replace=False)
    for idx in indexes:
        instance = X_test.iloc[[idx]]
        pred_start = time.perf_counter_ns()
        pred = clf.predict(instance)
        pred_end = time.perf_counter_ns()
        y_hat.append(pred)
        y_time.append(pred_end - pred_start)
    
    y_hat_batch = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_hat_batch)
    precision = precision_score(y_test, y_hat_batch, average="macro", zero_division=0)
    recall = recall_score(y_test, y_hat_batch, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_hat_batch, average="macro", zero_division=0)
    res = {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
        "indiv_pred_time": float(np.mean(y_time)),
        "p99_pred_time": float(np.percentile(y_time, 99)),
    }
    return res

def train_test_rust_dt(dataset_dir, feature_set, params):
    """
    Trains and tests Rust decision tree model using same train/test split and best hyperparameters.
    :param dataset_dir: str directory containing files of raw collected datasets
    :param feature_set: list feature set
    :param params: best hyperparameters found during Python tuning
    :return: dict containing Rust model performance attributes
    """
    print(utils.CYAN + "Training Rust DT" + utils.RESET)
    train_dataset_csv = os.path.join(dataset_dir, "train_dataset.csv")
    test_dataset_csv = os.path.join(dataset_dir, "test_dataset.csv")

    feature_decimal = utils.feature_decimal(feature_set)
    model_dir = os.path.join(dataset_dir, f"features_{feature_decimal}")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_pred_json = os.path.join(model_dir, "rust_dt_pred.json")
    model_bin = os.path.join(model_dir, "rust_dt.bin")

    max_depth = params["max_depth"]
    manifest_path = f"{consts.rust_train_dir}/train_rust_dt/Cargo.toml"

    feature_comma = utils.feature_comma(feature_set)

    # Rust DT trains the model using best hyperparameters found by CV in Python, 
    # and outputs its predictions on the test set with inference timings.
    cmd = f"cargo run --manifest-path={manifest_path} --release -- --train-dataset={train_dataset_csv} --test-dataset={test_dataset_csv} --feature-comma={feature_comma} --model-pred={model_pred_json} --model-bin={model_bin} --param-max-depth={max_depth}"

    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line, end="")
    
    return eval_rust_model(model_pred_json)

def eval_rust_model(model_pred_json):
    """
    Retrieves predictions and timing measurements of Rust classifier.
    :param model_pred_json: path to rust model predictions
    :return: dict containing model performance attributes
    """
    print(utils.CYAN + "Evaluating Rust DT" + utils.RESET)
    with open(model_pred_json) as file:
        model_pred = json.load(file)
    y_test = model_pred["y_test"]
    y_pred = model_pred["y_hat"]
    y_times = model_pred["y_time"]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    res = {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
        "indiv_pred_time": float(np.mean(y_times)),
        "p99_pred_time": float(np.percentile(y_times, 99)),
    }
    return res