import argparse
import os
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE

from helper import consts
from model_train_test import dataset_preprocess_iot
from model_train_test import dataset_preprocess_app
from model_train_test import dataset_preprocess_startup

def mi_feature_select(candidate_features, num_features, pkt_depth="all", dataset_dir=None):
    if not dataset_dir:
        dataset_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}")
    if consts.use_case == "app":
        df_train, _ = dataset_preprocess_app.preprocess_and_split(dataset_dir, pkt_depth)
        target = "label"
    elif consts.use_case == "iot":
        df_train, _ = dataset_preprocess_iot.preprocess_and_split(dataset_dir, pkt_depth)
        target = "label"
    elif consts.use_case == "startup":
        df_train, _ = dataset_preprocess_startup.preprocess_and_split(dataset_dir, pkt_depth)
        target = "startup_delay"
    else:
        raise ValueError(f"Invalid use case: {consts.use_case}")
    X_train = df_train[candidate_features]
    y_train = df_train[target]

    start_ts = time.time()
    if consts.use_case == "startup":
        mi_scores = mutual_info_regression(X_train, y_train, random_state=0)
    else:    
        mi_scores = mutual_info_classif(X_train, y_train, random_state=0)
    print(f"MI time elapsed: {time.time() - start_ts}s")

    mi_dict = {}
    for i, feature in enumerate(X_train.columns):
        mi_dict[feature] = mi_scores[i]
    sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_mi[:num_features]

def rfe_feature_select(candidate_features, num_features, pkt_depth="all", dataset_dir=None):
    if not dataset_dir:
        dataset_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}")
    if consts.use_case == "app":
        df_train, _ = dataset_preprocess_app.preprocess_and_split(dataset_dir, pkt_depth)
    elif consts.use_case == "iot":
        df_train, _ = dataset_preprocess_iot.preprocess_and_split(dataset_dir, pkt_depth)
    elif consts.use_case == "startup":
        df_train, _ = dataset_preprocess_startup.preprocess_and_split(dataset_dir, pkt_depth)
    else:
        raise ValueError(f"Invalid use case: {consts.use_case}")
    X_train = df_train[candidate_features]

    if consts.use_case == "startup":
        target = "startup_delay"
        y_train = df_train[target]
        clf = RandomForestRegressor()  # sklearn-compatible regressor for vid-start
    else:
        target = "label"
        y_train = df_train[target]
        if consts.model_type == "dt":
            clf = DecisionTreeClassifier()
        elif consts.model_type == "rf":
            clf = RandomForestClassifier()
        else:
            raise ValueError(f"Invalid model type: {consts.model_type}")
    rfe = RFE(clf, n_features_to_select=num_features)
    start_ts = time.time()
    rfe.fit(X_train, y_train)
    print(f"RFE time elapsed: {time.time() - start_ts}s")

    ranked_features = rfe.ranking_
    print(ranked_features)
    selected_features = rfe.support_ 
   
    rfe_features = []
    for i, feature in enumerate(X_train.columns):
        if selected_features[i]:
            rfe_features.append(feature)
    return rfe_features
