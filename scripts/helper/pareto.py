import numpy as np
import pandas as pd
import copy
from paretoset import paretoset
from pymoo.indicators.hv import HV

from measure import measure_compute
from measure import measure_inference

def is_pareto_min(costs):
    """
    Get min pareto efficient mask from costs.
    :param costs: a (n_points, 2) float array
    :return: a (n_points, ) boolean array mask indicating whether each cost is min pareto efficient
    """
    mask = paretoset(costs, sense=["min", "min"])
    return mask

def get_pareto_min(samples, perf_metric, cost_metric):
    """
    Get min pareto efficient points from samples.
    :param samples: DataFrame containing samples to compute pareto
    :return: DataFrame containing min pareto-efficient samples
    """
    mask = is_pareto_min(samples[[perf_metric, cost_metric]])
    return samples[mask]

def dominates(a, b):
    """
    Returns true if A dominates B
    :param a: First (feature_set, pkt_depth) tuple
    :param b: Second (feature_set, pkt_depth) tuple
    """
    a_feature_set, a_pkt_depth = a
    b_feature_set, b_pkt_depth = b
    a_neg_f1_score = -measure_inference.get_f1_score(a_feature_set, a_pkt_depth)
    a_compute_cost = measure_compute.get_compute_cost(a_feature_set, a_pkt_depth)
    b_neg_f1_score = -measure_inference.get_f1_score(b_feature_set, b_pkt_depth)
    b_compute_cost = measure_compute.get_compute_cost(b_feature_set, b_pkt_depth)
    return (a_neg_f1_score <= b_neg_f1_score and a_compute_cost <= b_compute_cost) and (a_neg_f1_score < b_neg_f1_score or a_compute_cost < b_compute_cost)

def get_max_cost(costs, percentile=100):
    """
    Get maximum cost to determine reference point for HVI computation
    TODO: exclude outliers.
    :param costs: a (n_points, ) float array of costs
    :return: maximum cost
    """
    return np.percentile(costs, percentile)

def compare_hvi(samples_dict, perf_metric, cost_metric, max_neg_f1=0, max_cost_percentile=100):
    """
    Get standardized HVI to compare estimated pareto fronts, normalized to real pareto front.
    :param samples_dict: dict of DataFrames containing samples to compare
    :param max_neg_f1: default 0, maximum negative F1 score to use as reference
    :param max_cost_percentile: default 100, maximum cost percentile to use as reference
    :return: dict of HVIs, mapping scheme to hvi
    """
    
    samples_dict = copy.deepcopy(samples_dict)
    # ground truth is the concat of all provided samples
    gt_samples = pd.concat(list(samples_dict.values()))
    max_compute_cost = np.max(gt_samples[cost_metric])
    gt_samples[cost_metric] = gt_samples[cost_metric] / max_compute_cost

    gt_neg_f1_scores = gt_samples[perf_metric].to_numpy()
    gt_compute_costs = gt_samples[cost_metric].to_numpy()
    
    # get max cost point for reference and standardize
    
    reference_point = np.array([
        # 0,
        max_neg_f1,
        get_max_cost(gt_compute_costs, max_cost_percentile),
    ])

    # standardize gt_pareto
    gt_pareto_samples = get_pareto_min(gt_samples, perf_metric, cost_metric).copy()
    gt_pareto_neg_f1_scores = gt_pareto_samples[perf_metric].to_numpy()
    gt_pareto_compute_costs = gt_pareto_samples[cost_metric].to_numpy()
    gt_pareto_points = np.array(list(zip(gt_pareto_neg_f1_scores, gt_pareto_compute_costs)))

    hvi_dict = {}
    for scheme, samples in samples_dict.items():
        samples[cost_metric] = samples[cost_metric] / max_compute_cost
        pareto_samples = get_pareto_min(samples, perf_metric, cost_metric).copy()
        pareto_neg_f1_scores = pareto_samples[perf_metric].to_numpy()
        pareto_compute_costs = pareto_samples[cost_metric].to_numpy()
        pareto_points = np.array(list(zip(pareto_neg_f1_scores, pareto_compute_costs)))

        hvi = get_hvi(pareto_points, gt_pareto_points, reference_point)
        hvi_dict[scheme] = hvi
    return hvi_dict
    

def get_hvi(pareto_points, gt_pareto_points, ref_point):
    """
    Get the hypervolume indicator of estimated pareto front normalized to that of the real pareto front.
    :param pareto_points: a (n_points, 2) float array of standardized costs in the pareto front
    :param gt_pareto_points: a (m_points, 2) float array of standardized costs in the ground truth pareto front
    :param ref_point: a (2, ) array used as the standardized reference point for computing HVI
    :return: float normalized HVI
    """
    hv_ind = HV(ref_point=ref_point)
    hvi = hv_ind(pareto_points)
    gt_hvi = hv_ind(gt_pareto_points)
    return hvi / gt_hvi
    
