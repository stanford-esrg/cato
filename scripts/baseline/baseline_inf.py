import os
import json
import time
import datetime

from helper import utils
from helper import consts
from helper import feature_selection
from measure import measure_compute
from measure import measure_inference
from measure import measure_throughput
from measure import measure_latency

def run_baseline_all():
    res = {}
    for pkt_depth in [10, 50,"all"]:
        feature_set = consts.candidate_features

        f1_score = measure_inference.get_f1_score(feature_set, pkt_depth)
        compute_cost = measure_compute.get_compute_cost(feature_set, pkt_depth)
        latency_cost = measure_latency.get_e2e_latency(feature_set, pkt_depth)
        throughput_cost = measure_throughput.get_throughput_cost(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "f1_score": f1_score,
            "compute_cost": compute_cost,
            "latency_cost": latency_cost,
            "throughput_cost": throughput_cost,
        }
    return res

def run_baseline_rfe():
    res = {}
    for pkt_depth in [10, 50,"all"]:
        rfe_features = feature_selection.rfe_feature_select(consts.candidate_features, 10, pkt_depth=pkt_depth)
        feature_set = sorted(rfe_features)

        f1_score = measure_inference.get_f1_score(feature_set, pkt_depth)
        compute_cost = measure_compute.get_compute_cost(feature_set, pkt_depth)
        latency_cost = measure_latency.get_e2e_latency(feature_set, pkt_depth)
        throughput_cost = measure_throughput.get_throughput_cost(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "f1_score": f1_score,
            "compute_cost": compute_cost,
            "latency_cost": latency_cost,
            "throughput_cost": throughput_cost,
        }
    return res

def run_baseline_mi():
    res = {}
    for pkt_depth in [10, 50,"all"]:
        mi_features = feature_selection.mi_feature_select(consts.candidate_features, 10, pkt_depth=pkt_depth)
        feature_set = sorted([k for k,v in mi_features])

        f1_score = measure_inference.get_f1_score(feature_set, pkt_depth)
        compute_cost = measure_compute.get_compute_cost(feature_set, pkt_depth)
        latency_cost = measure_latency.get_e2e_latency(feature_set, pkt_depth)
        throughput_cost = measure_throughput.get_throughput_cost(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "f1_score": f1_score,
            "compute_cost": compute_cost,
            "latency_cost": latency_cost,
            "throughput_cost": throughput_cost,
        }
    return res

def main():

    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    baseline_dir = os.path.join(consts.results_dir, "baselines")

    baselines = {}
    baselines["all"] = run_baseline_all()
    baselines["rfe"] = run_baseline_rfe()
    baselines["mi"] = run_baseline_mi()
    
    outfile = os.path.join(baseline_dir, f"baseline_{consts.model_type}_{ts}.json")
    with open(outfile, "w") as f:
        json.dump(baselines, f)
    

if __name__ == "__main__":
    main()
