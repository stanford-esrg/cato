import os
import json
import pickle

from helper import consts
from helper import utils
from model_train_test import dataset_preprocess_app
from model_train_test import dataset_preprocess_iot
from model_train_test import decision_tree
from model_train_test import random_forest


def measure_inference(feature_set, pkt_depth, model_type=consts.model_type, use_case=consts.use_case, dataset_dir=None):
    if not feature_set or (isinstance(pkt_depth, int) and pkt_depth < 1):
        print(utils.RED + "Empty feature set or 0 pkt depth" + utils.RESET)
        return 0
    if not dataset_dir:
        dataset_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}")
    feature_decimal = utils.feature_decimal(feature_set)
    model_dir = os.path.join(dataset_dir, f'features_{feature_decimal}')
    res_file = os.path.join(model_dir, f"inference_stats_{model_type}.json")
    if os.path.exists(res_file):
        with open(res_file, "r") as file:
            res = json.load(file)
        return res

    if use_case == "app":
        df_train, df_test = dataset_preprocess_app.preprocess_and_split(dataset_dir, pkt_depth)
    elif use_case == "iot":
        df_train, df_test = dataset_preprocess_iot.preprocess_and_split(dataset_dir, pkt_depth)
    else:
        raise ValueError(f"Invalid use case: {use_case}")
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    X_train = df_train[feature_set]
    y_train = df_train['label']
    if model_type == "dt":
        cv_res = decision_tree.train_python_dt(X_train, y_train)
    elif model_type == "rf":
        cv_res = random_forest.train_python_rf(X_train, y_train)
    else:
        raise ValueError(f"Invalid model type: {model_type}" + utils.RESET)
    
    X_test = df_test[feature_set]
    y_test = df_test['label']
    clf = cv_res['clf']
    params = cv_res['params']
    if model_type == "dt":
        python_res = decision_tree.test_python_dt(clf, X_test, y_test, 1000)
    elif model_type == "rf":
        python_res = random_forest.test_python_rf(clf, X_test, y_test, 1000)
    else:
        raise ValueError(f"Invalid model type: {model_type}" + utils.RESET)

    
    print(utils.GREEN + "Rust model results:" + utils.RESET)
    if model_type == "dt":
        rust_res = decision_tree.train_test_rust_dt(dataset_dir, feature_set, params)
    elif model_type == "rf":
        rust_res = random_forest.train_test_rust_rf(dataset_dir, feature_set, params)
    else:
        raise ValueError(f"Invalid model type: {model_type}" + utils.RESET)
    print(rust_res)

    del cv_res["clf"]
    res = {"cv": cv_res, "python": python_res, "rust": rust_res}

    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 
    with open(res_file, "w") as file:
        json.dump(res, file)
    with open(os.path.join(model_dir, f"python_{model_type}.pkl"), "wb") as file:
        pickle.dump(clf, file)
    return res

def get_f1_score(feature_set, pkt_depth):
    if not feature_set or pkt_depth == 0:
        return 0
    inference_stats = measure_inference(feature_set, pkt_depth, consts.model_type, consts.use_case)
    f1 = inference_stats["rust"]["f1"]
    return f1

def get_model_inf_time(feature_set, pkt_depth):
    if not feature_set or pkt_depth == 0:
        return 0
    inference_stats = measure_inference(feature_set, pkt_depth, consts.model_type, consts.use_case)
    # smartcore not parallelizing trees, approx for RF estimators
    inf_time = inference_stats["rust"]["indiv_pred_time"] / 100 
    return inf_time


