import argparse
import os
import json
import time
import datetime


from helper import utils
from helper import consts
from measure import measure_compute
from measure import measure_inference


def run_baseline_traffic_refinery_packet_counters():
    res = {}

    for pkt_depth in [10, 50,"all"]:
        feature_set = sorted(consts.packet_counters)

        f1_score = measure_inference.get_f1_score(feature_set, pkt_depth)
        compute_cost = measure_compute.get_compute_cost(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "f1_score": f1_score,
            "compute_cost": compute_cost,
        }
    return res

def run_baseline_traffic_refinery_packet_counters_packet_times():
    res = {}

    for pkt_depth in [10, 50,"all"]:
        feature_set = sorted(consts.packet_counters + consts.packet_times)

        f1_score = measure_inference.get_f1_score(feature_set, pkt_depth)
        compute_cost = measure_compute.get_compute_cost(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "f1_score": f1_score,
            "compute_cost": compute_cost,
        }
    return res

def run_baseline_traffic_refinery_packet_counters_packet_times_tcp_counters():
    res = {}

    for pkt_depth in [10, 50,"all"]:
        feature_set = sorted(consts.packet_counters + consts.packet_times + consts.tcp_counters)

        f1_score = measure_inference.get_f1_score(feature_set, pkt_depth)
        compute_cost = measure_compute.get_compute_cost(feature_set, pkt_depth)
        res[f"pkts_{pkt_depth}"] = {
            "feature_set": utils.feature_comma(feature_set), 
            "feature_decimal": utils.feature_decimal(feature_set),
            "f1_score": f1_score,
            "compute_cost": compute_cost,
        }
    return res


def main():
    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    baseline_dir = os.path.join(consts.results_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    baselines = {}
    baselines["packet_counters"] = run_baseline_traffic_refinery_packet_counters()
    baselines["packet_counters_packet_times"] = run_baseline_traffic_refinery_packet_counters_packet_times()
    baselines["packet_counters_packet_times_tcp_counters"] = run_baseline_traffic_refinery_packet_counters_packet_times_tcp_counters()
    
    outfile = os.path.join(baseline_dir, f"baseline_tr_{consts.model_type}_{ts}.json")
    with open(outfile, "w") as f:
        json.dump(baselines, f)

    
if __name__ == "__main__":
    main()
