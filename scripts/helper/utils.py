import hashlib
import itertools
import time
import random
import numpy as np

from helper import consts

# Color code escape sequences
RESET = '\033[0m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'

def feature_comma(feature_set):
    """
    Returns comma-delimited feature set in sorted order.
    :param feature_set: list of feature names
    :return: str comma-delimited features
    """
    feature_set = sorted(feature_set)
    return ','.join(feature_set)

def feature_hash(feature_set):
    """
    Returns SHA1 hash of sorted feature set dash delimited.
    :param feature_set: list of feature names
    :return: str SHA1 feature hash
    """
    feature_set = sorted(feature_set)
    feature_dash = '-'.join(feature_set)
    sha1 = hashlib.sha1()
    sha1.update(feature_dash.encode('utf-8'))
    feature_hash = sha1.hexdigest()
    return feature_hash

def feature_decimal(feature_set):
    """
    Returns a bitmask representation of feature set.
    :param feature_set: list of feature names
    :return: int decimal representation
    """
    active_idx = 0
    for feature in feature_set:
        assert feature in consts.candidate_features, f"{feature} not in candidate features"
        idx = consts.candidate_features.index(feature)
        active_idx |= 1 << idx
    return active_idx

def decimal_to_feature_set(decimal):
    """
    Converts bitmask decimal value to feature set based on index positions in candidate features.
    :param decimal: int decimal representation
    :return: list sorted feature set
    """
    feature_set = []
    for i, feature in enumerate(consts.candidate_features):
        if decimal & (1 << i):
            feature_set.append(feature)
    return sorted(feature_set)
            

def get_hash_to_feature_set(candidate_features):
    """
    Returns a dict mapping SHA1 hash to sorted list feature set.
    :param candidate_features: list of candidate features
    :return: dict mapping hash to sorted feature set
    """
    start_ts = time.time()
    feature_sets = get_all_feature_sets(candidate_features)
    res = {}
    for feature_set in feature_sets:
        res[feature_hash(feature_set)] = feature_set
    end_ts = time.time()
    print(f"Elapsed: {end_ts - start_ts}s")
    return res


def get_rand_feature_set(candidate_features, rs):
    """
    Get a random feature set from candidate features.
    :param candidate_features: list of candidate features
    :param rs: RandomState
    :return: a sorted feature list
    """
    size = len(candidate_features)
    rand_mask = [rs.choice([True, False]) for _ in range(size)]
    while not any(rand_mask):
        rand_mask = [rs.choice([True, False]) for _ in range(size)]
    rand_feature_set = [ft for ft, m in zip(candidate_features, rand_mask) if m]
    return sorted(rand_feature_set)

def get_rand_feature_set_with_size(candidate_features, num_features, rs):
    """
    Get a random feature set from candidate features with a given size.
    :param candidate_features: list of candidate features
    :param num_features: size of feature set
    :param rs: RandomState
    :return: a sorted feature list
    """
    rand_feature_set = rs.choice(candidate_features, size=num_features, replace=False)
    return sorted(rand_feature_set)


def get_all_feature_flags(candidate_features):
    """
    Get list of binary vectors representing all feature set combinations.
    :param candidate_features: list of candidate features
    :return: list of feature flags, 1 means feature is included, 0 not
    """
    size = len(candidate_features)
    return list(itertools.product([0,1], repeat=size))

def get_all_feature_sets(candidate_features):
    """
    Get list of all sorted feature set combinations.
    :param candidate_features: list of candidate features
    :return: list of sorted feature sets
    """
    all_feature_flags = get_all_feature_flags(candidate_features)
    feature_sets = []
    for feature_flags in all_feature_flags:
        # remove empty feature set
        if sum(feature_flags) == 0:
            continue
        nonzero_idx = np.nonzero(feature_flags)[0].tolist()
        feature_set = [candidate_features[i] for i in nonzero_idx]
        feature_sets.append(sorted(feature_set))
    return feature_sets

def get_all_feature_representations(candidate_features, max_pkt_depth):
    """
    Get list of all feature representations.
    :param candidate_features: list of candidate features
    :max_pkt_depth: maximum packet depth
    """
    feature_reps = []
    feature_sets = get_all_feature_sets(candidate_features)
    for feature_set in feature_sets:
        for pkt_depth in range(1, max_pkt_depth + 1):
            feature_reps.append((feature_set, pkt_depth))
    return feature_reps
