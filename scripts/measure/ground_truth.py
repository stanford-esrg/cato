import os
import json
import time
import glob
import pandas as pd
from pprint import pprint

from helper import utils
from helper import consts
from measure import measure_compute
from measure import measure_inference

def get_historical_ground_truth(max_pkt_depth, model_type, candidate_features):
    """
    Get dict mapping (comma-delimited feature set, pkt_depth) tuple to (negative F1 score, compute cost) tuple using only pre-measured points.
    """
    ground_truth = {}
    cnt = 0
    for pkt_depth in range(1, max_pkt_depth+1):
        syscost_dir = os.path.join(consts.syscost_dir, f"pkts_{pkt_depth}")
        dataset_dir = os.path.join(consts.dataset_dir, f'pkts_{pkt_depth}')

        measured_computes = glob.glob(f"{syscost_dir}/compute_features_*.csv")
        measured_inferences = glob.glob(f"{dataset_dir}/features_*/inference_stats_{model_type}.json")

        for compute_file in measured_computes:
            feature_decimal = int(compute_file.split('.')[0].split('_')[-1])
            feature_set = utils.decimal_to_feature_set(feature_decimal)
            if set(feature_set).difference(candidate_features):
                continue

            inference_stats_file = os.path.join(dataset_dir, f'features_{feature_decimal}/inference_stats_{model_type}.json')
            if inference_stats_file in measured_inferences:
                df = pd.read_csv(compute_file)
                compute_cost = df[df['name'] == 'compute_ns']
                compute = compute_cost['p99'].values[0]

                with open(inference_stats_file, 'r') as file:
                    res = json.load(file)
                f1 = res['rust']['f1']

                feature_comma = utils.feature_comma(feature_set)
                cnt += 1
                print(f"{cnt} > {feature_comma}@{pkt_depth}: {f1}, {compute}")
                ground_truth[(feature_comma, pkt_depth)] = (-f1, compute)
    return ground_truth
       
    
def get_ground_truth_samples(max_pkt_depth, model_type, candidate_features=[]):
    """
    Saves and returns DataFrame of previously measured ground truth measurements.
    """
    subset_decimal = utils.feature_decimal(sorted(candidate_features))
    out_file = os.path.join(consts.results_dir, "ground_truth", f"truth_subset{subset_decimal}_{model_type}_{max_pkt_depth}.csv")
    print(candidate_features, max_pkt_depth)
    print(out_file)
    if not os.path.isfile(out_file):
        gt = get_historical_ground_truth(max_pkt_depth, model_type, candidate_features)

        gt_flattened = [(k[0], k[1], v[0], v[1]) for k,v in gt.items()]
        df = pd.DataFrame(gt_flattened, columns=['feature_set', 'pkt_depth', 'neg_f1_score', 'compute_cost'])
        df.to_csv(out_file, index=False)
        return df
    else:
        df = pd.read_csv(out_file)
        return df
    
def measure_everything(candidate_features, pkt_depth_range):
    """
    Measures the ground truth inference stats and compute cost for all combinations of feature sets
    at each packet depth.
    """
    cnt = 0
    for pkt_depth in pkt_depth_range:
        for feature_set in utils.get_all_feature_sets(candidate_features):
            cnt += 1
            print(utils.MAGENTA + f"Count: {cnt}")
            print(pkt_depth, feature_set)
            print(f"Feature decimal: {utils.feature_decimal(feature_set)}" + utils.RESET)
            compute = measure_compute.measure_compute(feature_set, pkt_depth, query="on_demand")
            pprint(compute)
            inference_stats = measure_inference.measure_inference(feature_set, pkt_depth)
            pprint(inference_stats, sort_dicts=False)

    return cnt

def main():
    feature_list = [
        'dur',
        's_bytes_sum',
        's_bytes_mean',
        's_pkt_cnt',
        's_load',
        's_iat_mean',
    ]

    start_ts = time.time()
    measure_everything(feature_list, list(range(1,51)))
    end_ts = time.time()
    print(f"Elapsed: {end_ts - start_ts}s")
    

if __name__ == "__main__":
    main()
