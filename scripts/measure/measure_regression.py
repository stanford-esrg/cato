import os
import json
import pickle

from helper import consts
from helper import utils
from model_train_test import dataset_preprocess_startup
from model_train_test import dnn
from sklearn.preprocessing import StandardScaler


def measure_regression(feature_set, pkt_depth, model_type=consts.model_type, use_case=consts.use_case, dataset_dir=None):
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
            return json.load(file)

    if use_case == "startup":
        df_train, df_test = dataset_preprocess_startup.preprocess_and_split(dataset_dir, pkt_depth)
    else:
        raise ValueError(f"Invalid use case: {use_case}")

    X_train = df_train[feature_set]
    y_train = df_train['startup_delay']
    X_test = df_test[feature_set]
    y_test = df_test['startup_delay']
    
    if model_type == "dnn":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        cv_res = dnn.train_python_dnn(X_train, y_train.to_numpy())
        reg = cv_res['reg']
        python_res = dnn.test_python_dnn(reg, X_test, y_test.to_numpy(), 1000)

    del cv_res["reg"]
    res = {"cv": cv_res, "python": python_res, "rust": None}

    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 
    with open(res_file, "w") as file:
        json.dump(res, file)
    with open(os.path.join(model_dir, f"python_{model_type}.pkl"), "wb") as file:
        pickle.dump(reg, file)
    return res

def get_mae(feature_set, pkt_depth):
    if not feature_set or pkt_depth == 0:
        return 0
    inference_stats = measure_regression(feature_set, pkt_depth, consts.model_type, consts.use_case)
    return inference_stats["python"]["mae"]

def get_rmse(feature_set, pkt_depth):
    if not feature_set or pkt_depth == 0:
        return 0
    inference_stats = measure_regression(feature_set, pkt_depth, consts.model_type, consts.use_case)
    return inference_stats["python"]["rmse"]


