
import numpy as np
import torch
from torch.utils.data import DataLoader

from mmcls.utils import get_root_logger

from mmfscil.datasets import MemoryDataset
from .metrics import compute_old_class_drift
from .partition import partition_dataset
from .schedulers import compute_fusion_weights, should_reflect


class EdgeCloudServer:
    def __init__(self, model, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.edge_cfg = cfg.get('edge_cloud', {})
        self.device = device
        self.logger = get_root_logger()
        self.prev_class_means = None
        self.history_feats = None
        self.history_labels = None
        self.reflect_count = 0

    def _select_state_dict(self, share_backbone, share_neck, share_head):
        state = self.model.state_dict()
        prefixes = []
        if share_backbone:
            prefixes.append('backbone.')
        if share_neck:
            prefixes.append('neck.')
        if share_head:
            prefixes.append('head.')
        if not prefixes:
            return {}
        return {k: v.detach().cpu().clone()
                for k, v in state.items()
                if any(k.startswith(p) for p in prefixes)}

    def _load_state_dict(self, state):
        if not state:
            return
        current = self.model.state_dict()
        current.update(state)
        self.model.load_state_dict(current, strict=False)

    def build_clients(self, dataset):
        num_clients = int(self.edge_cfg.get('num_clients', self.edge_cfg.get('num_edges', 1)))
        part_cfg = self.edge_cfg.get('partition', dict(type='iid'))
        seed = int(self.edge_cfg.get('seed', 0))
        subsets = partition_dataset(dataset, num_clients, part_cfg, seed=seed)
        clients = []
        from .client import EdgeClient
        for idx, subset in enumerate(subsets):
            client = EdgeClient(idx, subset, self.cfg, self.device, self.logger, seed + idx)
            client.attach_model(self.model)
            clients.append(client)
        return clients

    def _aggregate(self, local_states, weights):
        if not local_states:
            return
        agg_state = {}
        for key in local_states[0].keys():
            agg_state[key] = torch.zeros_like(local_states[0][key])
        for weight, state in zip(weights, local_states):
            for key, value in state.items():
                agg_state[key] += value * float(weight)
        self._load_state_dict(agg_state)

    def _mutual_evaluation(self, clients, eval_batches):
        num_clients = len(clients)
        matrix = np.zeros((num_clients, num_clients), dtype=np.float64)
        for i, client in enumerate(clients):
            for j, batch in enumerate(eval_batches):
                result = client.evaluate_on_batch(batch)
                matrix[i, j] = result.get('acc', 0.0)
        return matrix

    def _compute_weights(self, matrix, client_stats):
        score_mode = self.edge_cfg.get('mutual_eval', {}).get('score', 'row_mean')
        if score_mode == 'col_mean':
            scores = matrix.mean(axis=0)
        else:
            scores = matrix.mean(axis=1)
        if self.edge_cfg.get('use_data_size', True):
            scores = scores * np.log1p(np.array([c.get('num_samples', 1) for c in client_stats]))
        if self.edge_cfg.get('use_entropy', True):
            ent = np.array([c.get('label_entropy', 1.0) for c in client_stats])
            if ent.max() > 0:
                scores = scores * (ent / ent.max())
        if self.edge_cfg.get('use_stability', True):
            stab = np.array([c.get('stability', 1.0) for c in client_stats])
            scores = scores * stab
        if self.edge_cfg.get('use_drift', True):
            drift = np.array([c.get('drift_mean', 0.0) for c in client_stats])
            scores = scores * np.exp(-drift)

        weight_cfg = self.edge_cfg.get('weighting', self.edge_cfg.get('fusion', {}))
        strategy = weight_cfg.get('strategy', 'softmax')
        beta = weight_cfg.get('beta', 5.0)
        cap = weight_cfg.get('cap', None)
        weights = compute_fusion_weights(scores, strategy=strategy, beta=beta, cap=cap)
        return weights, scores

    def _project_features(self, feats):
        x = feats.to(self.device)
        if hasattr(self.model, 'neck') and self.model.with_neck:
            x = self.model.neck(x)
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'pre_logits'):
            x = self.model.head.pre_logits(x)
        return x

    def _collect_memory(self, eval_batches):
        feats_list = []
        labels_list = []
        self.model.eval()
        for batch in eval_batches:
            if batch['mode'] == 'features':
                feats_list.append(batch['feat'])
                labels_list.append(batch['gt_label'])
            else:
                with torch.no_grad():
                    img = batch['img'].to(self.device)
                    feats = self.model.extract_feat(img, stage='backbone')
                feats_list.append(feats.detach().cpu())
                labels_list.append(batch['gt_label'])
        self.model.train()
        if not feats_list:
            return None, None
        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return feats, labels

    def _update_history(self, feats, labels):
        max_mem = int(self.edge_cfg.get('reflect', {}).get('max_memory', 2048))
        if self.history_feats is None:
            self.history_feats = feats.clone()
            self.history_labels = labels.clone()
        else:
            self.history_feats = torch.cat([self.history_feats, feats], dim=0)
            self.history_labels = torch.cat([self.history_labels, labels], dim=0)
        if self.history_feats.size(0) > max_mem:
            self.history_feats = self.history_feats[-max_mem:]
            self.history_labels = self.history_labels[-max_mem:]

    def _get_history(self):
        return self.history_feats, self.history_labels

    def _compute_class_means(self, feats, labels, num_classes):
        labels = labels.to(feats.device)
        means = []
        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                mean_feat = feats[mask].mean(dim=0, keepdim=True)
                means.append(mean_feat)
        if not means:
            return None
        return torch.cat(means, dim=0)

    def _reflect_alignment(self, memory_feats, memory_labels):
        reflect_cfg = self.edge_cfg.get('reflect', {})
        if not reflect_cfg.get('enabled', False):
            return
        if memory_feats is None or memory_labels is None:
            return

        steps = int(reflect_cfg.get('steps', 20))
        batch_size = int(reflect_cfg.get('batch_size', 32))
        lr = float(reflect_cfg.get('lr', 0.01))
        margin = float(reflect_cfg.get('margin', 0.1))
        lambda_reflect = float(reflect_cfg.get('lambda_reflect', 1.0))
        neg_topk = int(reflect_cfg.get('neg_topk', 1))

        dataset = MemoryDataset(memory_feats, memory_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        params = []
        if hasattr(self.model, 'neck') and self.model.neck is not None:
            params += list(self.model.neck.parameters())
        if hasattr(self.model, 'head') and self.model.head is not None:
            params += list(self.model.head.parameters())
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

        self.model.train()
        loader_iter = iter(loader)
        for _ in range(steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            feats = batch['feat'].to(self.device)
            labels = batch['gt_label'].to(self.device)
            optimizer.zero_grad()
            proj_feat = self._project_features(feats)
            if self.model.head.with_len:
                etf_vec = self.model.head.etf_vec * self.model.head.etf_rect.to(device=self.device)
                target = (etf_vec * self.model.head.produce_training_rect(labels, self.model.head.num_classes))[
                    :, labels].t()
                m_norm2 = torch.norm(target, p=2, dim=1)
                loss_dr = self.model.head.compute_loss(proj_feat, target, m_norm2=m_norm2)
            else:
                target = self.model.head.etf_vec[:, labels].t()
                loss_dr = self.model.head.compute_loss(proj_feat, target)

            logits = proj_feat @ self.model.head.etf_vec
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(labels.size(0)), labels] = False
            neg_logits = logits.masked_fill(~mask, -1e9)
            if neg_topk > 1:
                neg_vals, _ = torch.topk(neg_logits, k=neg_topk, dim=1)
                neg_score = neg_vals.mean(dim=1)
            else:
                neg_score = torch.max(neg_logits, dim=1)[0]
            pos_score = logits[torch.arange(labels.size(0)), labels]
            loss_margin = torch.relu(margin - pos_score + neg_score).mean()
            loss = loss_dr + lambda_reflect * loss_margin
            loss.backward()
            optimizer.step()

    def run_rounds(self,
                   clients,
                   input_mode,
                   global_rounds,
                   local_epochs,
                   optimizer_cfg,
                   share_backbone,
                   share_neck,
                   share_head,
                   train_backbone,
                   log_prefix='base'):
        self.reflect_count = 0
        global_state = self._select_state_dict(share_backbone, share_neck, share_head)
        reflect_cfg = self.edge_cfg.get('reflect', {})
        drift_threshold = float(reflect_cfg.get('threshold', reflect_cfg.get('drift_threshold', 1.0)))

        for rnd in range(global_rounds):
            local_states = []
            client_stats = []
            eval_batches = []
            for client in clients:
                client.set_trainable(train_backbone, True, True)
                local_state, _, stats, eval_batch = client.train_local(
                    global_state,
                    input_mode=input_mode,
                    local_epochs=local_epochs,
                    optimizer_cfg=optimizer_cfg,
                    share_backbone=share_backbone,
                    share_neck=share_neck,
                    share_head=share_head,
                    grad_clip=self.cfg.get('grad_clip', None),
                    log_interval=self.edge_cfg.get('log_interval', 50),
                )
                local_states.append(local_state)
                client_stats.append(stats)
                eval_batches.append(eval_batch)

            mutual_cfg = self.edge_cfg.get('mutual_eval', {})
            if mutual_cfg.get('enabled', True):
                matrix = self._mutual_evaluation(clients, eval_batches)
            else:
                matrix = np.eye(len(clients), dtype=np.float64)
            weights, scores = self._compute_weights(matrix, client_stats)
            self._aggregate(local_states, weights)
            global_state = self._select_state_dict(share_backbone, share_neck, share_head)

            memory_feats, memory_labels = self._collect_memory(eval_batches)
            if memory_feats is not None:
                self._update_history(memory_feats, memory_labels)
                proj_feats = self._project_features(memory_feats)
                class_means = self._compute_class_means(proj_feats, memory_labels, self.model.head.num_classes)
                old_drift = compute_old_class_drift(self.prev_class_means, class_means)
                self.prev_class_means = class_means
            else:
                old_drift = 0.0

            drift_mean = float(np.mean([s.get('drift_mean', 0.0) for s in client_stats]))
            do_reflect = should_reflect(drift_mean, drift_threshold)
            if do_reflect:
                self.logger.info('[EdgeCloud] Reflect triggered at round {} ({:.4f} > {:.4f}).'.format(
                    rnd + 1, drift_mean, drift_threshold))
                self.reflect_count += 1
                hist_feats, hist_labels = self._get_history()
                self._reflect_alignment(hist_feats, hist_labels)

            self.logger.info('[EdgeCloud][{}][round {}] M shape {} scores {} alpha {} sum {:.4f} drift {:.4f} old {:.4f}'.format(
                log_prefix,
                rnd + 1,
                matrix.shape,
                np.round(scores, 4).tolist(),
                np.round(weights, 4).tolist(),
                float(weights.sum()) if weights.size > 0 else 0.0,
                drift_mean,
                old_drift,
            ))
        self.logger.info('[EdgeCloud][{}] Reflect count {}'.format(log_prefix, self.reflect_count))
