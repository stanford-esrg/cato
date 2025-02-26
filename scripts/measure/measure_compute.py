import os
import pandas as pd

from helper import utils
from helper import consts
from measure import retina


def measure_compute(feature_set, pkt_depth, duration=60, query="on_demand"):
    """
    Measure compute cost for given feature set and packet depth.
    :param feature_set: list of feature names
    :param pkt_depth: int representing maximum packet depth features are collected at
    :param query:
        on_demand: invoke retina if previous measurement does not exist
        always: always invoke retina
        never: never invoke retina - return None if does not exist
    :return: float compute cost in nanoseconds ranging in [0,inf), or None if cannot retrieve
    """
    if not feature_set or (isinstance(pkt_depth, int) and pkt_depth < 1):
        print(utils.RED + "Empty feature set or 0 pkt depth" + utils.RESET)
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
    compute_cost = df[df['name'] == 'compute_ns']
    return {"avg": float(compute_cost['avg'].values[0]), "p50": float(compute_cost['p50'].values[0]), "p99": float(compute_cost['p99'].values[0])}
    
def get_compute_cost(feature_set, pkt_depth):
    """
    Retrieve p99 execution time.
    """
    if not feature_set or str(pkt_depth) == "0":
        return 0
    res = measure_compute(feature_set, pkt_depth, query="on_demand")
    cost = res["p99"]
    return cost
    
