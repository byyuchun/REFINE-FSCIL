import math

import numpy as np
import torch


def class_distribution_entropy(labels, num_classes=None):
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels).astype(np.int64)
    if labels.size == 0:
        return 0.0
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    probs = counts / (counts.sum() + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy)


def _normalize(tensor, dim=1, eps=1e-12):
    return tensor / (tensor.norm(p=2, dim=dim, keepdim=True) + eps)


def compute_proto_drift(feats, labels, proto_vec, eps=1e-12):
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    feats = feats.float()
    labels = labels.long()
    proto_vec = proto_vec.float()
    feats = _normalize(feats, dim=1, eps=eps)
    proto_vec = _normalize(proto_vec, dim=0, eps=eps)
    logits = feats @ proto_vec
    pos = logits[torch.arange(labels.size(0)), labels]
    drift = 1.0 - pos
    return {
        'drift_mean': float(drift.mean().item()),
        'drift_p90': float(torch.quantile(drift, 0.9).item()),
        'drift_max': float(drift.max().item()),
    }


def compute_old_class_drift(prev_means, curr_means, eps=1e-12):
    if prev_means is None or curr_means is None:
        return 0.0
    if prev_means.numel() == 0 or curr_means.numel() == 0:
        return 0.0
    prev_means = _normalize(prev_means, dim=1, eps=eps)
    curr_means = _normalize(curr_means, dim=1, eps=eps)
    cos = torch.sum(prev_means * curr_means, dim=1)
    drift = 1.0 - cos
    return float(drift.mean().item())


def compute_stability(prev_loss, cur_loss):
    if prev_loss is None:
        return 1.0
    return float(math.exp(-abs(prev_loss - cur_loss)))
