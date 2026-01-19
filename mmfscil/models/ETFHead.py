import json
import math
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from mmcv.runner import get_dist_info
from mmcls.utils import get_root_logger

from mmcls.models.heads import ClsHead

from mmcls.models.builder import HEADS, LOSSES


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


@LOSSES.register_module()
class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight


@HEADS.register_module()
class ETFHead(ClsHead):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int, *args, **kwargs) -> None:
        # Pre-assign fixed targets to avoid fine-tuning conflicts that induce
        # forgetting; "pre-assign and fix an optimal feature-classifier
        # alignment ... to avoid target conflict" (ICLR 2023 core idea).
        proto_mode = kwargs.pop('proto_mode', 'ETF')
        sim_cfg = kwargs.pop('sim_cfg', None)
        self.proto_mode = str(proto_mode).upper()
        if self.proto_mode not in ['ETF', 'SIM']:
            raise ValueError(f'proto_mode={self.proto_mode} is not supported')
        self.sim_cfg = sim_cfg if sim_cfg is not None else {}

        if kwargs.get('eval_classes', None):
            self.eval_classes = kwargs.pop('eval_classes')
        else:
            self.eval_classes = num_classes

        # training settings about different length for different classes
        if kwargs.pop('with_len', False):
            self.with_len = True
        else:
            self.with_len = False

        super().__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels
        logger = get_root_logger()
        logger.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))
        logger.info("ETF head : with_len : {}".format(self.with_len))
        logger.info("Proto mode : {}".format(self.proto_mode))

        # ETF: maximal angular separation with a uniform simplex structure.
        # SIM: encode inter-class relations from a semantic similarity matrix
        # into fixed targets, yielding similarity-aligned prototypes.
        if self.proto_mode == 'ETF':
            etf_vec = self._build_etf_proto()
        else:
            etf_vec = self._build_sim_proto(logger)
        self.register_buffer('etf_vec', etf_vec)
        self.etf_vec.requires_grad_(False)
        self._proto_synced = False

        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect
        self._log_proto_stats(logger)

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def forward_train(self, x: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> Dict:
        """Forward training data."""
        self._maybe_sync_proto()
        x = self.pre_logits(x)
        if self.with_len:
            etf_vec = self.etf_vec * self.etf_rect.to(device=self.etf_vec.device)
            target = (etf_vec * self.produce_training_rect(gt_label, self.num_classes))[:, gt_label].t()
        else:
            target = self.etf_vec[:, gt_label].t()
        losses = self.loss(x, target)
        if self.cal_acc:
            with torch.no_grad():
                cls_score = x @ self.etf_vec
                acc = self.compute_accuracy(cls_score[:, :self.eval_classes], gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        return losses

    def mixup_extra_training(self, x: torch.Tensor) -> Dict:
        self._maybe_sync_proto()
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        assigned = torch.argmax(cls_score[:, self.eval_classes:], dim=1)
        target = self.etf_vec[:, assigned + self.eval_classes].t()
        losses = self.loss(x, target)
        return losses

    def loss(self, feat, target, **kwargs):
        losses = dict()
        # compute loss
        if self.with_len:
            loss = self.compute_loss(feat, target, m_norm2=torch.norm(target, p=2, dim=1))
        else:
            loss = self.compute_loss(feat, target)
        losses['loss'] = loss
        return losses

    def simple_test(self, x, softmax=False, post_process=False):
        self._maybe_sync_proto()
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes]
        assert not softmax
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score

    @staticmethod
    def produce_training_rect(label: torch.Tensor, num_classes: int):
        rank, world_size = get_dist_info()
        if world_size > 0:
            recv_list = [None for _ in range(world_size)]
            dist.all_gather_object(recv_list, label.cpu())
            new_label = torch.cat(recv_list).to(device=label.device)
            label = new_label
        uni_label, count = torch.unique(label, return_counts=True)
        batch_size = label.size(0)
        uni_label_num = uni_label.size(0)
        assert batch_size == torch.sum(count)
        gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
        rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
        rect[0, uni_label] = torch.sqrt(gamma / count)
        return rect

    def _build_etf_proto(self) -> torch.Tensor:
        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        return etf_vec

    def _build_sim_proto(self, logger) -> torch.Tensor:
        # SIM prototypes are generated by spectral embedding of a similarity
        # matrix, with a robust fallback optimization when rank/conditioning
        # is insufficient for the target dimensionality.
        sim_cfg = self._get_sim_cfg()
        sim_eps = float(sim_cfg['sim_eps'])
        eig_tol = float(sim_cfg['eig_tol'])
        sim_path = sim_cfg.get('sim_path')
        if sim_path:
            logger.info("SIM proto : loading similarity matrix from {}.".format(sim_path))
        sim_mat = self._load_similarity_matrix(sim_cfg, logger)
        sim_mat = self._sanitize_similarity(sim_mat, sim_eps)
        if sim_mat.shape != (self.num_classes, self.num_classes):
            raise ValueError("Similarity matrix shape mismatch: {} vs {}.".format(
                sim_mat.shape, (self.num_classes, self.num_classes)))

        w_np = self._spectral_embed(sim_mat, self.in_channels, sim_eps, eig_tol)
        if w_np is None:
            logger.warning("SIM proto : spectral embedding failed or rank < d, using fallback optimization.")
            w_t = self._fallback_optimize(sim_mat, sim_cfg, logger)
            w_np = w_t.cpu().numpy()

        w_np = self._hungarian_align(w_np, sim_mat, sim_cfg, logger)
        w_np = self._normalize_columns_np(w_np, sim_eps)
        return torch.from_numpy(w_np).float()

    def _get_sim_cfg(self) -> Dict:
        cfg = dict(
            sim_path=None,
            sim_format=None,
            sim_eps=1e-6,
            eig_tol=1e-6,
            hungarian=False,
            hungarian_max_classes=200,
            fallback_steps=100,
            fallback_lr=0.1,
        )
        if self.sim_cfg:
            cfg.update(self.sim_cfg)
        return cfg

    def _load_similarity_matrix(self, sim_cfg: Dict, logger) -> np.ndarray:
        sim_path = sim_cfg.get('sim_path')
        if not sim_path:
            logger.info("SIM proto : no similarity matrix provided, using identity.")
            return np.eye(self.num_classes, dtype=np.float64)
        if not os.path.isfile(sim_path):
            raise FileNotFoundError("Similarity matrix not found: {}".format(sim_path))

        sim_format = sim_cfg.get('sim_format')
        if sim_format:
            sim_format = sim_format.lower()
            if not sim_format.startswith('.'):
                sim_format = '.' + sim_format
        else:
            sim_format = os.path.splitext(sim_path)[1].lower()

        if sim_format == '.npy':
            sim_mat = np.load(sim_path)
        elif sim_format == '.json':
            with open(sim_path, 'r', encoding='utf-8') as handle:
                sim_mat = np.array(json.load(handle), dtype=np.float64)
        else:
            raise ValueError("Unsupported similarity matrix format: {}".format(sim_format))
        return sim_mat

    @staticmethod
    def _sanitize_similarity(sim_mat: np.ndarray, eps: float) -> np.ndarray:
        sim_mat = np.asarray(sim_mat, dtype=np.float64)
        if sim_mat.ndim != 2 or sim_mat.shape[0] != sim_mat.shape[1]:
            raise ValueError("Similarity matrix must be square.")
        sim_mat = np.clip(sim_mat, 0.0, 1.0)
        sim_mat = 0.5 * (sim_mat + sim_mat.T)
        sim_mat = sim_mat + eps * np.eye(sim_mat.shape[0])
        sim_mat = 0.5 * (sim_mat + sim_mat.T)
        return sim_mat

    @staticmethod
    def _normalize_columns_np(mat: np.ndarray, eps: float) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        return mat / (norms + eps)

    @staticmethod
    def _normalize_columns_torch(mat: torch.Tensor, eps: float) -> torch.Tensor:
        return mat / (mat.norm(dim=0, keepdim=True) + eps)

    @staticmethod
    def _spectral_embed(sim_mat: np.ndarray, d: int, eps: float, eig_tol: float) -> Optional[np.ndarray]:
        try:
            eigvals, eigvecs = np.linalg.eigh(sim_mat)
        except np.linalg.LinAlgError:
            return None
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        rank = int(np.sum(eigvals > eig_tol))
        if rank < d:
            return None
        vals = np.maximum(eigvals[:d], 0.0)
        emb = eigvecs[:, :d] * np.sqrt(vals)
        w_np = emb.T
        w_np = ETFHead._normalize_columns_np(w_np, eps)
        if not np.isfinite(w_np).all():
            return None
        return w_np

    def _fallback_optimize(self, sim_mat: np.ndarray, sim_cfg: Dict, logger) -> torch.Tensor:
        steps = int(sim_cfg.get('fallback_steps', 100))
        lr = float(sim_cfg.get('fallback_lr', 0.1))
        eps = float(sim_cfg.get('sim_eps', 1e-6))
        target = 2.0 * sim_mat - 1.0
        target = torch.from_numpy(target).float()
        proto = torch.randn(self.in_channels, self.num_classes, dtype=torch.float32)
        proto.requires_grad_(True)
        optimizer = torch.optim.SGD([proto], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            gram = proto.t() @ proto
            loss = torch.mean((gram - target) ** 2)
            loss.backward()
            optimizer.step()
        proto = proto.detach()
        proto = self._normalize_columns_torch(proto, eps)
        if not torch.isfinite(proto).all():
            logger.warning("SIM proto : fallback produced non-finite values, using random normalized.")
            proto = self._normalize_columns_torch(
                torch.randn(self.in_channels, self.num_classes, dtype=torch.float32), eps)
        return proto

    def _hungarian_align(self, w_np: np.ndarray, sim_mat: np.ndarray, sim_cfg: Dict, logger) -> np.ndarray:
        # Hungarian alignment solves a global one-to-one assignment between
        # classes and fixed prototypes, reducing arbitrary associations that
        # can introduce target conflict.
        if not sim_cfg.get('hungarian', False):
            return w_np
        max_classes = int(sim_cfg.get('hungarian_max_classes', 200))
        if self.num_classes > max_classes:
            logger.warning("SIM proto : Hungarian alignment skipped (C={} > {}).".format(
                self.num_classes, max_classes))
            return w_np
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as exc:
            logger.warning("SIM proto : scipy not available for Hungarian ({})".format(exc))
            return w_np

        # Cost matches each class row in S to a prototype row in W^T W.
        proto_sim = w_np.T @ w_np
        diff = sim_mat[:, None, :] - proto_sim[None, :, :]
        cost = np.mean(diff ** 2, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(self.num_classes, dtype=np.int64)
        perm[row_ind] = col_ind
        return w_np[:, perm]

    def _maybe_sync_proto(self) -> None:
        if self.proto_mode != 'SIM' or self._proto_synced:
            return
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.etf_vec, src=0)
        self._proto_synced = True

    def _log_proto_stats(self, logger) -> None:
        norms = torch.norm(self.etf_vec, dim=0)
        logger.info(
            "Proto stats : shape={} requires_grad={} norm_mean={:.4f} norm_var={:.6f}".format(
                tuple(self.etf_vec.shape),
                self.etf_vec.requires_grad,
                norms.mean().item(),
                norms.var(unbiased=False).item(),
            )
        )
