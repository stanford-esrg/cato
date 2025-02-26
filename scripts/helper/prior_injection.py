
from helper import feature_selection

def compute_mi_scores(candidate_features, pkt_depth):
    mi_scores = feature_selection.mi_feature_select(candidate_features, len(candidate_features), pkt_depth=pkt_depth)
    return dict(mi_scores)

def compute_normalized_mi_scores(candidate_features, pkt_depth):
    feature_mi = feature_selection.mi_feature_select(candidate_features, len(candidate_features), pkt_depth=pkt_depth)
    values = [x[1] for x in feature_mi]
    max_value = max(values)
    normalized_mi = [(x[0], x[1] / max_value) for x in feature_mi]
    return dict(normalized_mi)


def compute_utilities(candidate_features, pkt_depth):
    """
    Computes utility for each candidate feature.
    """
    mi_scores = compute_normalized_mi_scores(candidate_features, pkt_depth)
    utilities = {}
    for feature in candidate_features:
        w = 1
        utility = w * mi_scores[feature] #+ (1-w) * cost_scores[feature]
        utilities[feature] = utility
    return utilities

def dampen_utility(util, damping_factor=0):
    """
    Dampens utility.
    :param util: Utility value, which is the probability of inclusion.
    :param damping_factor: Scales the utility value. Damp=1 returns 0.5, Damp = 0 returns util
    """
    if util < 0 or util > 1 or damping_factor < 0 or damping_factor > 1:
        raise ValueError("Util or damping factor out of range")

    if damping_factor == 0:
        return util
    if damping_factor == 1:
        return 0.5
    return (1 - damping_factor) * util + damping_factor * 0.5


def compute_priors(candidate_features, pkt_depth, damping_factor=0):
    """
    Computes prior for each candidate feature and packet depth.
    :param candidate_features: list of candidate features
    :param pkt_depth: pkt depth to compute feature priors at
    :param damping_factor: 0 means no damping, 1 means there should be no prior placed over features (0.5 probability of inclusion)
    """

    utilities = compute_utilities(candidate_features, pkt_depth)
    print(utilities)
    priors = {}
    for feature, utility in utilities.items():
        p = dampen_utility(utility, damping_factor)
        prior = [1-p, p]
        priors[feature] = prior

    return priors


