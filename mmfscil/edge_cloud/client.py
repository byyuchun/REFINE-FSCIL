import copy

import numpy as np
import torch
from mmcv.parallel import DataContainer

from mmcls.datasets import build_dataloader
from mmcv.runner import build_optimizer

from .metrics import compute_proto_drift, class_distribution_entropy, compute_stability


def _unwrap(data):
    if isinstance(data, DataContainer):
        data = data.data
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    return data


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    return x


class EdgeClient:
    def __init__(self, client_id, dataset, cfg, device, logger, seed=0):
        self.client_id = client_id
        self.dataset = dataset
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.seed = seed
        self.model = None
        self.prev_loss = None

        edge_cfg = cfg.get('edge_cloud', {})
        self.local_batch_size = edge_cfg.get('local_batch_size', None)
        self.eval_batch_size = edge_cfg.get('eval_batch_size', 32)
        self.eval_share = edge_cfg.get('eval_share', 'features')

        self.train_loader = self._build_train_loader()

    def attach_model(self, model):
        self.model = copy.deepcopy(model).to(self.device)

    def set_trainable(self, train_backbone=True, train_neck=True, train_head=True):
        if hasattr(self.model, 'backbone') and self.model.backbone is not None:
            for param in self.model.backbone.parameters():
                param.requires_grad = bool(train_backbone)
        if hasattr(self.model, 'neck') and self.model.neck is not None:
            for param in self.model.neck.parameters():
                param.requires_grad = bool(train_neck)
        if hasattr(self.model, 'head') and self.model.head is not None:
            for param in self.model.head.parameters():
                param.requires_grad = bool(train_head)

    def _build_train_loader(self):
        loader_cfg = dict(
            num_gpus=1,
            dist=False,
            round_up=True,
            seed=self.cfg.get('seed', None),
            sampler_cfg=self.cfg.get('sampler', None),
        )
        loader_cfg.update({
            k: v
            for k, v in self.cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })
        train_loader_cfg = {**loader_cfg, **self.cfg.data.get('train_dataloader', {})}
        if self.local_batch_size is not None:
            train_loader_cfg['samples_per_gpu'] = self.local_batch_size
        return build_dataloader(self.dataset, **train_loader_cfg)

    def _extract_batch(self, data, input_mode):
        if input_mode == 'images':
            img = _unwrap(data.get('img', None))
            gt_label = _unwrap(data.get('gt_label', None))
            img = _to_device(img, self.device)
            gt_label = _to_device(gt_label, self.device)
            if torch.is_tensor(gt_label) and gt_label.dim() > 1:
                gt_label = gt_label.view(-1)
            return img, gt_label
        feat = _unwrap(data.get('feat', None))
        gt_label = _unwrap(data.get('gt_label', None))
        feat = _to_device(feat, self.device)
        gt_label = _to_device(gt_label, self.device)
        if torch.is_tensor(gt_label) and gt_label.dim() > 1:
            gt_label = gt_label.view(-1)
        return feat, gt_label

    def _forward_features(self, feats):
        x = feats
        if hasattr(self.model, 'neck') and self.model.with_neck:
            x = self.model.neck(x)
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'pre_logits'):
            x = self.model.head.pre_logits(x)
        return x

    def _forward_head(self, feats):
        x = feats
        if hasattr(self.model, 'neck') and self.model.with_neck:
            x = self.model.neck(x)
        return self.model.head.simple_test(x, softmax=False, post_process=False)

    def _get_state_dict(self, share_backbone, share_neck, share_head):
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
        filtered = {k: v.detach().cpu().clone()
                    for k, v in state.items()
                    if any(k.startswith(p) for p in prefixes)}
        return filtered

    def _load_state_dict(self, global_state):
        if not global_state:
            return
        state = self.model.state_dict()
        state.update(global_state)
        self.model.load_state_dict(state, strict=False)

    def sample_eval_batch(self, input_mode):
        data_iter = iter(self.train_loader)
        data = next(data_iter)
        img_or_feat, gt_label = self._extract_batch(data, input_mode)
        if img_or_feat.size(0) > self.eval_batch_size:
            img_or_feat = img_or_feat[:self.eval_batch_size]
            gt_label = gt_label[:self.eval_batch_size]
        if self.eval_share == 'images' and input_mode == 'images':
            return {
                'mode': 'images',
                'img': img_or_feat.detach().cpu(),
                'gt_label': gt_label.detach().cpu(),
            }
        self.model.eval()
        with torch.no_grad():
            if input_mode == 'images':
                feats = self.model.extract_feat(img_or_feat, stage='backbone')
            else:
                feats = img_or_feat
        self.model.train()
        return {
            'mode': 'features',
            'feat': feats.detach().cpu(),
            'gt_label': gt_label.detach().cpu(),
        }

    def evaluate_on_batch(self, batch):
        mode = batch.get('mode', 'features')
        with torch.no_grad():
            if mode == 'images':
                img = _to_device(batch['img'], self.device)
                gt_label = _to_device(batch['gt_label'], self.device)
                if torch.is_tensor(gt_label) and gt_label.dim() > 1:
                    gt_label = gt_label.view(-1)
                res = self.model(return_loss=False, return_acc=True, img=img, gt_label=gt_label)
                acc = float(np.mean(res))
                return {'acc': acc}
            feats = _to_device(batch['feat'], self.device)
            gt_label = _to_device(batch['gt_label'], self.device)
            if torch.is_tensor(gt_label) and gt_label.dim() > 1:
                gt_label = gt_label.view(-1)
            cls_score = self._forward_head(feats)
            pred = torch.argmax(cls_score, dim=1)
            acc = torch.mean((pred == gt_label).float()).item()
            return {'acc': float(acc)}

    def train_local(self,
                    global_state,
                    input_mode,
                    local_epochs,
                    optimizer_cfg,
                    share_backbone,
                    share_neck,
                    share_head,
                    grad_clip=None,
                    log_interval=50):
        self._load_state_dict(global_state)
        eval_batch = self.sample_eval_batch(input_mode)

        if optimizer_cfg is None:
            raise ValueError('optimizer_cfg must be provided')
        optimizer = build_optimizer(self.model, optimizer_cfg)

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        self.model.train()
        for epoch in range(local_epochs):
            for idx, data in enumerate(self.train_loader):
                img_or_feat, gt_label = self._extract_batch(data, input_mode)
                optimizer.zero_grad()
                if input_mode == 'images':
                    losses = self.model(return_loss=True, img=img_or_feat, gt_label=gt_label)
                else:
                    feats = img_or_feat
                    if hasattr(self.model, 'neck') and self.model.with_neck:
                        feats = self.model.neck(feats)
                    losses = self.model.head.forward_train(feats, gt_label)
                loss = losses['loss']
                loss.backward()
                if grad_clip is not None:
                    params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                optimizer.step()

                batch_size = gt_label.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                acc = None
                if 'accuracy' in losses and 'top-1' in losses['accuracy']:
                    acc = losses['accuracy']['top-1'].item()
                if acc is not None:
                    total_acc += acc * batch_size
                if (idx + 1) % log_interval == 0:
                    self.logger.info(
                        '[Edge {}] epoch {} step {} loss {:.4f}'.format(
                            self.client_id, epoch + 1, idx + 1, loss.item()))

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_acc / max(total_samples, 1) if total_acc > 0 else 0.0

        stats = {
            'loss': float(avg_loss),
            'acc': float(avg_acc),
            'num_samples': int(total_samples),
        }

        if eval_batch['mode'] == 'features':
            feats = eval_batch['feat'].to(self.device)
            gt_label = eval_batch['gt_label'].to(self.device)
            with torch.no_grad():
                proj_feat = self._forward_features(feats)
            drift = compute_proto_drift(proj_feat, gt_label, self.model.head.etf_vec)
            stats.update(drift)

        labels = []
        if hasattr(self.dataset, 'data_infos'):
            labels = [int(info.get('cls_id', info.get('gt_label', 0))) for info in self.dataset.data_infos]
        elif hasattr(self.dataset, 'labels'):
            labels = self.dataset.labels.detach().cpu().numpy().tolist()
        if labels:
            stats['label_entropy'] = class_distribution_entropy(labels, num_classes=self.model.head.num_classes)
        stats['stability'] = compute_stability(self.prev_loss, avg_loss)
        self.prev_loss = avg_loss

        local_state = self._get_state_dict(share_backbone, share_neck, share_head)
        delta_state = {}
        for key, value in local_state.items():
            if key in global_state:
                delta_state[key] = value - global_state[key]
        return local_state, delta_state, stats, eval_batch
