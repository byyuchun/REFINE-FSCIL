import numpy as np


def compute_fusion_weights(scores,
                           strategy='softmax',
                           beta=5.0,
                           cap=None,
                           eps=1e-12):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.array([])
    if strategy == 'softmax':
        scaled = scores * float(beta)
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        weights = weights / (weights.sum() + eps)
        return weights
    if strategy == 'capped':
        scaled = scores * float(beta)
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        weights = weights / (weights.sum() + eps)
        if cap is not None:
            cap = float(cap)
            weights = np.minimum(weights, cap)
            weights = weights / (weights.sum() + eps)
        return weights
    if strategy == 'uniform':
        return np.ones_like(scores, dtype=np.float64) / (scores.size + eps)
    raise ValueError('Unsupported weight strategy: {}'.format(strategy))


def should_reflect(drift_value, threshold):
    return float(drift_value) > float(threshold)
