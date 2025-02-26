import os
import json
import time
import datetime

from helper import utils
from helper import consts
from helper import feature_selection
from measure import measure_regression
from measure import measure_latency

def run_baseline_all():
    res = {}
    for pkt_depth in [10, 50,"all"]:
        print(f"all, pkt_depth{pkt_depth}")
        feature_set = consts.candidate_features

        rmse = measure_regression.get_rmse(feature_set, pkt_depth)
        mae = measure_regression.get_mae(feature_set, pkt_depth)
        latency_cost = measure_latency.get_inference_latency(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "rmse": rmse,
            "mae": mae,
            "latency_cost": latency_cost,
        }
    return res

def run_baseline_rfe():
    """
    This baseline does recursive feature elimination for the top 10 features. 
    :return: A result dict.
    """
    res = {}
    for pkt_depth in [10, 50,"all"]:
        print(f"rfe, pkt_depth{pkt_depth}")
        rfe_features = feature_selection.rfe_feature_select(consts.candidate_features, 10, pkt_depth=pkt_depth)
        feature_set = sorted(rfe_features)

        rmse = measure_regression.get_rmse(feature_set, pkt_depth)
        mae = measure_regression.get_mae(feature_set, pkt_depth)
        latency_cost = measure_latency.get_inference_latency(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "rmse": rmse,
            "mae": mae,
            "latency_cost": latency_cost,
        }
    return res


def run_baseline_mi():
    res = {}
    for pkt_depth in [10, 50,"all"]:
        print(f"mi, pkt_depth{pkt_depth}")
        mi_features = feature_selection.mi_feature_select(consts.candidate_features, 10, pkt_depth=pkt_depth)
        feature_set = sorted([k for k,v in mi_features])

        rmse = measure_regression.get_rmse(feature_set, pkt_depth)
        mae = measure_regression.get_mae(feature_set, pkt_depth)
        latency_cost = measure_latency.get_inference_latency(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "rmse": rmse,
            "mae": mae,
            "latency_cost": latency_cost,
        }
    return res

def main():
    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    baseline_dir = os.path.join(consts.results_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    baselines = {}
    baselines["all"] = run_baseline_all()
    baselines["rfe"] = run_baseline_rfe()
    baselines["mi"] = run_baseline_mi()
    
    outfile = os.path.join(baseline_dir, f"baseline_{consts.model_type}_{ts}.json")
    with open(outfile, "w") as f:
        json.dump(baselines, f)

    

if __name__ == "__main__":
    main()
