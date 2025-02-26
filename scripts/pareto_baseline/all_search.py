import os
import argparse
import csv
import time
import datetime
import sys

from helper import utils
from helper import consts
from measure import measure_compute
from measure import measure_inference


def as_run(candidate_features, max_pkt_depth, num_iter, experiment_dir=""):
    """
    Run all search optimization and save results.
    :param candidate_features: list of candidate features
    :param max_pkt_depth: int maximum packet depth to consider
    :param num_iter: int number of optimization iterations
    :param experiment_dir: directory to put results
    :return: str output results file path
    """

    # create new timestamped output directory
    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    all_search_dir = os.path.join(consts.results_dir, "all_search", experiment_dir)
    if not os.path.exists(all_search_dir):
        os.mkdir(all_search_dir)
    output_dir = os.path.join(all_search_dir, f"max{max_pkt_depth}_iter{num_iter}_{ts}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    headers = ["feature_set", "pkt_depth", "neg_f1_score", "compute_cost", "Timestamp"]
    output_file = os.path.join(output_dir, "as_output_samples.csv")
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
        start_ts = time.time_ns() / 1e6
        for iter in range(num_iter):
            print(f"Starting all search iteration {iter}")
            feature_set = candidate_features
            pkt_depth = iter + 1
            f1_score = measure_inference.get_f1_score(feature_set, pkt_depth)
            compute_cost = measure_compute.get_compute_cost(feature_set, pkt_depth)

            ts = time.time_ns() / 1e6 - start_ts
            feature_comma = utils.feature_comma(feature_set)
            row_data = [feature_comma, pkt_depth, -f1_score, compute_cost, ts]
            print(row_data)
            writer.writerow(row_data)
            file.flush()

    return output_file

def main(args):
    candidate_features = [
        'dur',
        's_bytes_sum',
        's_bytes_mean',
        's_pkt_cnt',
        's_load',
        's_iat_mean',
    ]

    print(utils.CYAN + f"Candidate features: {candidate_features}" + utils.RESET)
    print(utils.CYAN + f"Number of candidate features: {len(candidate_features)}" + utils.RESET)

    for i in range(args.num_trials):
        print(f"Trial {i+1}")
        start_ts = time.time()
        as_run(candidate_features, args.max_pkt_depth, args.num_iter, experiment_dir=args.experiment_dir)
        end_ts = time.time()
        print(f"All search elapsed: {end_ts - start_ts}s")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize pareto front using all search."
    )
    parser.add_argument("max_pkt_depth", type=int, help="Maximum packet depth")
    parser.add_argument("num_iter", type=int, help="Number of all search iterations")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment output dir")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    main(parser.parse_args())
