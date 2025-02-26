import os
from natsort import natsorted
import json
import time


from helper import utils
from helper import consts
from measure import retina
from measure import measure_inference


def measure_throughput(feature_set, pkt_depth, duration):
    """
    Retrieve zero-loss throughput for given feature set and packet depth.
    :param feature_set: list of feature names
    :param pkt_depth: int representing maximum packet depth features are collected at
    :param duration: int number of seconds to run each trial
    :return: path to results directory
    """
    # check if model directory exists
    feature_decimal = utils.feature_decimal(feature_set)
    model_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}", f"features_{feature_decimal}")
    if not os.path.exists(model_dir):
        print(utils.CYAN + "Model directory does not exist, training..." + utils.RESET)
        measure_inference.measure_inference(feature_set, pkt_depth)

    # decrease RSS buckets until zero-loss is reached
    bucket_range = list(range(512,0,-32)) + [1]
    return retina.measure_throughput(feature_set, pkt_depth, bucket_range, duration)
    
def get_throughput_cost(feature_set, pkt_depth):
    start = time.time()
    pkts_fts_ts_throughput_dir = measure_throughput(feature_set, pkt_depth, 15)
    feature_decimal = utils.feature_decimal(feature_set)

    zero_loss_rate = 0
    for path in list_dirs(pkts_fts_ts_throughput_dir):
        try:
            outfile = os.path.join(path, f"out_features_{feature_decimal}.json")
            with open(outfile, "r") as file:
                out = json.load(file)
            processed_rate = out["num_conns"] / out["config"]["online"]["duration"]

            log_dir = list_dirs(path)[0]
            statsfile = os.path.join(log_dir, "throughputs.json")
            with open(statsfile, "r") as file:
                stats = json.load(file)
            dropped = stats["percent_dropped"]
            if dropped == 0:
                zero_loss_rate = max(processed_rate, zero_loss_rate)
        except:
            continue
    elapsed = time.time() - start
    print(utils.MAGENTA + f"tput cost time: {elapsed} s" + utils.RESET)
    # negate for cost function
    return -zero_loss_rate

def list_dirs(dir_path):
    dirs = [os.path.join(dir_path, name) for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    return natsorted(dirs)


