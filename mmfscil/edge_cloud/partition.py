import numpy as np
from torch.utils.data import Subset


def _safe_int(val):
    try:
        return int(val)
    except Exception:
        try:
            return int(np.asarray(val).item())
        except Exception:
            return int(val[0])


def _get_labels(dataset):
    if hasattr(dataset, 'data_infos'):
        labels = []
        for info in dataset.data_infos:
            if 'gt_label' in info:
                labels.append(_safe_int(info['gt_label']))
            elif 'cls_id' in info:
                labels.append(_safe_int(info['cls_id']))
        if labels:
            return np.array(labels, dtype=np.int64)
    if hasattr(dataset, 'get_gt_labels'):
        labels = dataset.get_gt_labels()
        return np.array(labels, dtype=np.int64)
    return None


def _build_subset(dataset, indices):
    subset = Subset(dataset, indices)
    if hasattr(dataset, 'CLASSES'):
        subset.CLASSES = dataset.CLASSES
    return subset


def partition_dataset(dataset, num_clients, partition_cfg, seed=0):
    if num_clients <= 0:
        raise ValueError('num_clients must be positive')

    part_type = str(partition_cfg.get('type', 'iid')).lower()
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))

    if part_type == 'iid':
        rng.shuffle(indices)
        ratios = partition_cfg.get('ratios')
        if ratios:
            ratios = np.array(ratios, dtype=np.float64)
            ratios = ratios / ratios.sum()
        else:
            ratios = np.ones(num_clients, dtype=np.float64) / num_clients
        splits = (ratios * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()
        subsets = []
        start = 0
        for count in splits:
            end = start + count
            subsets.append(_build_subset(dataset, indices[start:end]))
            start = end
        return subsets

    if part_type == 'by_class':
        labels = _get_labels(dataset)
        if labels is None:
            rng.shuffle(indices)
            splits = np.array_split(indices, num_clients)
            return [_build_subset(dataset, split) for split in splits]
        classes = np.unique(labels)
        rng.shuffle(classes)
        class_per_client = partition_cfg.get('classes_per_client')
        if class_per_client is None:
            class_per_client = int(np.ceil(len(classes) / float(num_clients)))
        client_classes = [[] for _ in range(num_clients)]
        for idx, cls in enumerate(classes):
            client_classes[idx // class_per_client].append(cls)
        subsets = []
        for cls_list in client_classes:
            mask = np.isin(labels, cls_list)
            subsets.append(_build_subset(dataset, indices[mask]))
        return subsets

    raise ValueError('Unsupported partition type: {}'.format(part_type))
