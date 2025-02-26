import os
import argparse
import pandas as pd
import time
import csv
import json
import jsonlines
import numpy as np

from helper import utils
from helper import consts
from measure import retina
from measure import measure_compute
from measure import measure_regression
from measure import measure_inference

def get_total_iat(pkt_depth, dataset_dir=None):
    """
    Fetch total interarrival time of packets up to pkt_depth.
    """
    if str(pkt_depth) == "0" or str(pkt_depth) == "1":
        return 0
    if consts.use_case == "iot" or consts.use_case == "startup":
        if not dataset_dir:
            dataset_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}")

        raw_features_path = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}", "features.jsonl")
        raw_features_csv = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}", "raw_features.csv")
        if not os.path.exists(raw_features_csv):
            if not os.path.exists(raw_features_path):
                retina.collect_raw_dataset(pkt_depth, outfile_name="features.jsonl")

            first_row = True
            start_ts = time.time()
            with open(raw_features_csv, 'w', newline='') as writer:
                overall_cnt = 0
                with jsonlines.open(raw_features_path) as reader:
                    csv_writer = csv.writer(writer)
                    for obj in reader:
                        keys = list(obj.keys())
                        if first_row:
                            csv_writer.writerow(keys)
                            first_row = False
                        csv_writer.writerow(obj[key] for key in keys)
                        overall_cnt += 1
            end_ts = time.time()
            print(f"Rows: {overall_cnt}, elapsed: {end_ts - start_ts}s")
        raw_features = pd.read_csv(raw_features_csv)
    elif consts.use_case == "app":
        raw_features_csv = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}", "dataset.csv")
        raw_features = pd.read_csv(raw_features_csv)
    else:
        raise ValueError(f"Invalid use case: {consts.use_case}")
    
    iat_sum_lst = raw_features["s_iat_sum"].fillna(0) + raw_features["d_iat_sum"].fillna(0)
    if len(iat_sum_lst) == 0:
        return 0
    return np.mean(iat_sum_lst)

def get_inference_latency(feature_set, pkt_depth, model_type=consts.model_type, use_case=consts.use_case):
    """
    Measure e2e latency cost for a given feature set and packet depth (pcap traffic offline mode).
    Includes IAT, feature extraction time, and model inference (execution) time.
    """
    print(utils.CYAN, feature_set, pkt_depth, utils.RESET)
    iat_tot = get_total_iat(pkt_depth)
    if not feature_set or (isinstance(pkt_depth, int) and pkt_depth < 1):
        ft_extract_time = 0
        model_inf_time = 0
    else:
        ft_extract_time = measure_compute.get_compute_cost(feature_set, pkt_depth)
        if use_case == "startup":
            if model_type == "dnn":
                model_inf_time = measure_regression.measure_regression(feature_set, pkt_depth, model_type, use_case)["python"]["p99_pred_time"]
            else:
                model_inf_time = measure_regression.measure_regression(feature_set, pkt_depth, model_type, use_case)["rust"]["p99_pred_time"]
        else:
            model_inf_time = measure_inference.measure_inference(feature_set, pkt_depth, model_type, use_case)["rust"]["p99_pred_time"]
    inf_latency = iat_tot + ft_extract_time + model_inf_time
    return inf_latency / 1e9

def measure_e2e(feature_set, pkt_depth, duration=60, query="on_demand"):
    """
    Measure e2e latency cost for given feature set and packet depth (live traffic online mode).
    :param feature_set: list of feature names
    :param pkt_depth: int representing maximum packet depth features are collected at
    :param query:
        on_demand: invoke retina if previous measurement does not exist
        always: always invoke retina
        never: never invoke retina - return None if does not exist
    :return: float compute cost in nanoseconds ranging in [0,inf), or None if cannot retrieve
    """
    if not feature_set or (isinstance(pkt_depth, int) and pkt_depth < 1):
        return 0
    feature_decimal = utils.feature_decimal(feature_set)
    syscost_dir = os.path.join(consts.syscost_dir, f'pkts_{pkt_depth}')
    compute_file = os.path.join(syscost_dir, f'compute_features_{feature_decimal}.csv')
    if query == "always":
        if not retina.measure_costs(feature_set, pkt_depth, duration):
            print(utils.RED + "Retina failed to measure costs" + utils.RESET)
            return None
    elif query == "on_demand":
        if not os.path.exists(compute_file):
            if not retina.measure_costs(feature_set, pkt_depth, duration):
                print(utils.RED + "Retina failed to measure costs" + utils.RESET)
                return None
    elif query == "never":
        if not os.path.exists(compute_file):
            print(utils.RED + "Compute measurement does not exist." + utils.RESET)
            return None
    else:
        raise ValueError(f"Invalid query option: {query}.")
        
    df = pd.read_csv(compute_file)
    latency_cost = df[df['name'] == 'e2e_ns']
    return {"avg": float(latency_cost['avg'].values[0]), "p99": float(latency_cost['p99'].values[0])}

def get_e2e_latency(feature_set, pkt_depth):
    """
    Retrieve e2e inference latency (for live traffic/online mode). Use get_inference_latency() 
    for offline modes to compute IAT from pcap timestamps.
    :param feature_set: list of feature names
    :param pkt_depth: "all" or int representing maximum packet depth features are collected at
    """
    if not feature_set or str(pkt_depth) == "0":
        return 0
    res = measure_e2e(feature_set, pkt_depth, duration=120, query="on_demand")
    return res["avg"] / 1e9

def measure_all_latency(candidate_features, pkt_depth_range):
    """
    Measures the ground truth latency all combinations of feature sets
    at each packet depth.
    """
    if len(candidate_features) > 6:
        raise ValueError("Only run this for minified candidate set")
    if consts.use_case == "app":
        raise NotImplementedError
    elif consts.use_case == "iot":
        max_pkt_depth = max(pkt_depth_range)
        cnt = 0
        e2e_latency = {}
        outfile = os.path.join(consts.results_dir, "ground_truth", f"e2e_latency_{consts.model_type}_{max_pkt_depth}.json")
        for pkt_depth in pkt_depth_range:
            for feature_set in utils.get_all_feature_sets(candidate_features):
                cnt += 1
                print(utils.MAGENTA + f"Count: {cnt}")
                print(pkt_depth, feature_set)
                feature_decimal = utils.feature_decimal(feature_set)
                print(f"Feature decimal: {feature_decimal}" + utils.RESET)
                latency = get_inference_latency(feature_set, pkt_depth)
                feature_comma = utils.feature_comma(feature_set)
                key = f"{feature_comma}@{pkt_depth}"
                e2e_latency[key] = latency
        with open(outfile, "w") as f:
            json.dump(e2e_latency, f)
    else:
        raise ValueError("Invalid use case")
    

