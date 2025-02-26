import os
import argparse
import numpy as np
import csv
import time
import datetime
import random
import math

from helper import utils
from helper import consts
from helper import pareto
from measure import measure_compute
from measure import measure_inference

def initial_point(candidate_features, max_pkt_depth, rs):
    """
    Generate an initial feature representation at random.
    :param candidate_features: list of candidate features
    :param max_pkt_depth: int max packet depth to consider
    :param rs: random state
    :return: random feature representation tuple
    """
    feature_set = utils.get_rand_feature_set(candidate_features, rs)
    pkt_depth = rs.randint(1, max_pkt_depth + 1)
    return (feature_set, pkt_depth)

def get_neighbor(curr_point, candidate_features, max_pkt_depth, iteration, max_iterations, visited, rs):
    """
    Generate a neighbor feature representation that is close to the current rep.
    :param curr_point: current feature representation
    :param candidate_features: list of candidate features
    :param max_pkt_depth: int max packet depth to consider
    :param iteration: int current iteration
    :param max_iters: int maximum number of iterations
    :param visited: set of visited point
    :param rs: random state
    :return: random feature representation tuple
    """
    max_attempts = 10

    for _ in range(max_attempts):
        feature_set, pkt_depth = curr_point
        feature_set = set(feature_set)

        # perturb either feature set or packet depth at random
        if random.random() < 0.5:
            if len(feature_set) == 1:
                operation = rs.choice(["add", "replace"])
            elif len(feature_set) == len(candidate_features):
                operation = rs.choice(["remove", "replace"])
            else:
                operation = rs.choice(["add", "remove", "replace"])
            new_feature_set = feature_set.copy()

            if operation == "add":
                possible_additions = list(set(candidate_features) - set(feature_set))
                feature_to_add = rs.choice(possible_additions)
                new_feature_set.add(feature_to_add)
            elif operation == "remove":
                feature_to_remove = rs.choice(list(new_feature_set))
                new_feature_set.remove(feature_to_remove)
            elif operation == "replace":
                feature_to_remove = rs.choice(list(new_feature_set))
                new_feature_set.remove(feature_to_remove)
                possible_replacements = list(set(candidate_features) - set(new_feature_set))
                feature_to_add = rs.choice(possible_replacements)
                new_feature_set.add(feature_to_add)
            
            new_point = (sorted(new_feature_set), pkt_depth)
        else:
            progress = iteration / max_iterations
            max_step = max(1, int(max_pkt_depth * (1 - progress)))  # Larger steps at the start
            print(utils.CYAN, max_step, utils.RESET)
            step = rs.randint(-max_step, max_step)
            new_pkt_depth = max(1, min(max_pkt_depth, pkt_depth + step))
            new_point = (sorted(feature_set), new_pkt_depth)

        return new_point
    # return current point if a new point is not found
    return curr_point
    
def weighted_cost(point):
    """
    Get weighted cost of both f1 score and compute cost.
    :param point: feature representation
    :return: float combined cost
    """
    # prior domain knowledge needed to normalize cost to approx [0,1]
    est_max_compute_cost = 4000 
    norm_f1_score = -measure_inference.get_f1_score(point[0], point[1]) + 1
    norm_compute_cost = measure_compute.get_compute_cost(point[0], point[1]) / est_max_compute_cost
    return 0.5 * norm_f1_score + 0.5 * norm_compute_cost


def sa_run(candidate_features, max_pkt_depth, num_iter, init_temp, random_state=None, experiment_dir=""):
    """
    Run simulated annealing optimization and save results.
    :param candidate_features: list of candidate features
    :param max_pkt_depth: int maximum packet depth to consider
    :param num_iter: int number of optimization iterations
    :param experiment_dir: directory to put results
    :return: str output results file path
    """
    candidate_decimal = utils.feature_decimal(candidate_features)
    final_temp = 1e-3
    rs = np.random.RandomState(random_state)

    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    sa_dir = os.path.join(consts.results_dir, f"sa_{candidate_decimal}")
    if not os.path.exists(sa_dir):
        os.mkdir(sa_dir)
    exp_dir = os.path.join(sa_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    output_dir = os.path.join(exp_dir, f"max{max_pkt_depth}_iter{num_iter}_temp{init_temp}_{ts}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    headers = ["feature_set", "pkt_depth", "neg_f1_score", "compute_cost", "Timestamp"]
    output_file = os.path.join(output_dir, "sa_output_samples.csv")

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        start_ts = time.time_ns() / 1e6
        visited = set()
        curr_point = initial_point(candidate_features, max_pkt_depth, rs)

        feature_comma = utils.feature_comma(curr_point[0])
        f1_score = measure_inference.get_f1_score(curr_point[0], curr_point[1])
        compute_cost = measure_compute.get_compute_cost(curr_point[0], curr_point[1])
        ts = time.time_ns() / 1e6 - start_ts
        row_data = [feature_comma, curr_point[1], -f1_score, compute_cost, ts]
        print(row_data)
        writer.writerow(row_data)
        file.flush()
        current_temp = init_temp

        for iter in range(1, num_iter):
            print(utils.CYAN + f"Starting simulated annealing iteration {iter}, temp = {current_temp}" + utils.RESET)
            ts = time.time_ns() / 1e6 - start_ts
            new_point = get_neighbor(curr_point, candidate_features, max_pkt_depth, iter, num_iter, visited, rs)

            curr_f1_score = measure_inference.get_f1_score(curr_point[0], curr_point[1])
            curr_compute_cost = measure_compute.get_compute_cost(curr_point[0], curr_point[1])
            new_f1_score = measure_inference.get_f1_score(new_point[0], new_point[1])
            new_compute_cost = measure_compute.get_compute_cost(new_point[0], new_point[1])
            print(f"curr: {curr_point}: ({curr_f1_score}, {curr_compute_cost}), cost: {weighted_cost(curr_point)}")
            print(f"new:  {new_point}: ({new_f1_score}, {new_compute_cost}), cost: {weighted_cost(new_point)}")

            feature_comma = utils.feature_comma(new_point[0])
            row_data = [feature_comma, new_point[1], -new_f1_score, new_compute_cost, ts]
            print(row_data)
            writer.writerow(row_data)
            file.flush()

        
            if pareto.dominates(new_point, curr_point):
                print(utils.GREEN + "ACCEPT" + utils.RESET)
                curr_point = new_point
            elif not pareto.dominates(curr_point, new_point):
                print("curr_point does not dominate new_point")
                accept_probab = math.exp((weighted_cost(curr_point) - weighted_cost(new_point))/ current_temp)
                print(f"accept probab: {accept_probab}")
                if accept_probab > rs.random():
                    print(utils.YELLOW + "ACCEPT" + utils.RESET)
                    curr_point = new_point
                else:
                    print(utils.MAGENTA + "REJECT" + utils.RESET)
            else:
                print(utils.RED + "REJECT" + utils.RESET)

            current_temp = max(final_temp, 0.99 * current_temp)

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
        for init_temp in args.init_temp.split(","):
            start_ts = time.time()
            sa_run(candidate_features, args.max_pkt_depth, args.num_iter, int(init_temp), random_state=None, experiment_dir=args.experiment_dir)
            end_ts = time.time()
            print(f"Simulated annealing elapsed: {end_ts - start_ts}s")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize pareto front using simulated annealing."
    )
    parser.add_argument("max_pkt_depth", type=int, help="Maximum packet depth")
    parser.add_argument("num_iter", type=int, help="Number of simulated annealing iterations")
    parser.add_argument("init_temp", type=str, help="Initial temperature")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment output dir")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    main(parser.parse_args())
