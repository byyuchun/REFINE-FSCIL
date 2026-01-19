import copy
import os

import torch
from mmcv.runner import save_checkpoint

from mmcls.utils import get_root_logger
from mmfscil.apis.fscil import (
    get_training_memory,
    get_test_memory,
    get_inc_memory,
    test_session,
    test_session_feat,
)
from mmfscil.datasets import MemoryDataset

from .server import EdgeCloudServer


def edge_cloud_train(model,
                     datasets,
                     cfg,
                     distributed=False,
                     validate=False,
                     timestamp=None,
                     device=None,
                     meta=None):
    logger = get_root_logger()
    device = device or cfg.get('device', 'cuda')
    dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    train_dataset = dataset[0]

    edge_cfg = cfg.get('edge_cloud', {})
    server = EdgeCloudServer(model, cfg, device)
    clients = server.build_clients(train_dataset)

    global_rounds = int(edge_cfg.get('base_rounds', edge_cfg.get('global_rounds', 1)))
    local_epochs = int(edge_cfg.get('base_local_epochs', edge_cfg.get('local_epochs', 1)))
    share_backbone = bool(edge_cfg.get('share_backbone', True))
    share_neck = bool(edge_cfg.get('share_neck', True))
    share_head = bool(edge_cfg.get('share_head', True))
    train_backbone = bool(edge_cfg.get('train_backbone', True))
    optimizer_cfg = edge_cfg.get('local_optimizer', None)
    if optimizer_cfg is None:
        optimizer_cfg = cfg.optimizer

    logger.info('[EdgeCloud] Base training with {} clients, rounds {}, local epochs {}'.format(
        len(clients), global_rounds, local_epochs))
    server.run_rounds(
        clients,
        input_mode='images',
        global_rounds=global_rounds,
        local_epochs=local_epochs,
        optimizer_cfg=optimizer_cfg,
        share_backbone=share_backbone,
        share_neck=share_neck,
        share_head=share_head,
        train_backbone=train_backbone,
        log_prefix='base',
    )

    if cfg.work_dir:
        ckpt_path = os.path.join(cfg.work_dir, 'edge_cloud_base.pth')
        save_checkpoint(server.model, ckpt_path)
        logger.info('[EdgeCloud] Saved base checkpoint to {}'.format(ckpt_path))


def edge_cloud_fscil(model,
                     cfg,
                     distributed=False,
                     validate=False,
                     timestamp=None,
                     meta=None):
    logger = get_root_logger()
    device = cfg.get('device', 'cuda')

    inc_start = cfg.inc_start
    inc_end = cfg.inc_end
    inc_step = cfg.inc_step
    edge_cfg = cfg.get('edge_cloud', {})

    model_inf = copy.deepcopy(model).to(device)
    model_inf.eval()
    proto_memory, proto_memory_label = get_training_memory(cfg, model_inf, logger, distributed)
    test_feat, test_label = get_test_memory(cfg, model_inf, logger, distributed)
    inc_feat, inc_label = get_inc_memory(cfg, model_inf, logger, distributed, inc_start, inc_end)

    model_finetune = copy.deepcopy(model).to(device)
    if hasattr(model_finetune, 'backbone') and model_finetune.backbone is not None:
        for param in model_finetune.backbone.parameters():
            param.requires_grad = False

    acc_list = []
    acc = test_session(cfg, model_finetune, distributed, test_feat, test_label, logger, 1, 0, inc_start, inc_start)
    acc_list.append(acc)
    logger.info('[EdgeCloud] Start incremental sessions.')
    save_checkpoint(model_finetune, os.path.join(cfg.work_dir, 'session_0.pth'))

    for i in range((inc_end - inc_start) // inc_step):
        label_start = inc_start + i * inc_step
        label_end = inc_start + (i + 1) * inc_step
        logger.info('[EdgeCloud] Session {} classes {}-{}.'.format(i + 2, label_start, label_end))
        model_finetune.head.eval_classes = label_end

        num_steps = cfg.step_list[i]
        if num_steps > 0:
            if cfg.mean_cur_feat:
                mean_feat = []
                mean_label = []
                for idx in range(inc_start, label_end):
                    mean_feat.append(inc_feat[inc_label == idx].mean(dim=0, keepdim=True))
                    mean_label.append(inc_label[inc_label == idx][0:1])
                cur_session_feats = torch.cat(mean_feat).repeat(cfg.copy_list[i], 1, 1, 1)
                cur_session_labels = torch.cat(mean_label).repeat(cfg.copy_list[i])
            elif cfg.mean_neck_feat:
                cur_session_feats = inc_feat[
                    torch.logical_and(torch.ge(inc_label, label_start), torch.less(inc_label, label_end))]
                cur_session_labels = inc_label[
                    torch.logical_and(torch.ge(inc_label, label_start), torch.less(inc_label, label_end))]
                mean_feat = []
                mean_label = []
                for idx in range(inc_start, label_start):
                    mean_feat.append(inc_feat[inc_label == idx].mean(dim=0, keepdim=True))
                    mean_label.append(inc_label[inc_label == idx][0:1])
                if label_start > inc_start:
                    cur_session_feats = torch.cat(
                        [cur_session_feats, torch.cat(mean_feat).repeat(cfg.copy_list[i], 1, 1, 1)])
                    cur_session_labels = torch.cat(
                        [cur_session_labels, torch.cat(mean_label).repeat(cfg.copy_list[i])])
            else:
                cur_session_feats = inc_feat[
                    torch.logical_and(torch.ge(inc_label, inc_start), torch.less(inc_label, label_end))]
                cur_session_labels = inc_label[
                    torch.logical_and(torch.ge(inc_label, inc_start), torch.less(inc_label, label_end))]

            cur_session_feats = torch.cat([cur_session_feats, proto_memory], dim=0)
            cur_session_labels = torch.cat([cur_session_labels, proto_memory_label], dim=0)
            cur_dataset = MemoryDataset(feats=cur_session_feats, labels=cur_session_labels)

            server = EdgeCloudServer(model_finetune, cfg, device)
            clients = server.build_clients(cur_dataset)
            global_rounds = int(edge_cfg.get('inc_rounds', edge_cfg.get('global_rounds', 1)))
            local_epochs = int(edge_cfg.get('inc_local_epochs', edge_cfg.get('local_epochs', 1)))
            optimizer_cfg = edge_cfg.get('finetune_optimizer', None)
            if optimizer_cfg is None:
                optimizer_cfg = dict(
                    type='SGD', lr=cfg.finetune_lr, momentum=0.9, weight_decay=0.0005)
            server.run_rounds(
                clients,
                input_mode='features',
                global_rounds=global_rounds,
                local_epochs=local_epochs,
                optimizer_cfg=optimizer_cfg,
                share_backbone=False,
                share_neck=True,
                share_head=True,
                train_backbone=False,
                log_prefix='session_{}'.format(i + 2),
            )

            if cfg.feat_test:
                cls_feat = []
                for idx in range(0, label_end):
                    cls_feat.append(cur_session_feats[cur_session_labels == idx].mean(dim=0, keepdim=True))
                cls_feat = torch.cat(cls_feat)
                acc = test_session_feat(cfg, model_finetune, distributed, test_feat, test_label,
                                        cls_feat, logger, i + 2, 0, label_end, inc_start)
            else:
                acc = test_session(cfg, model_finetune, distributed, test_feat, test_label,
                                   logger, i + 2, 0, label_end, inc_start)
            acc_list.append(acc)
            save_checkpoint(model_finetune, os.path.join(cfg.work_dir, 'session_{}.pth'.format(i + 1)))

    acc_str = ''
    for acc in acc_list:
        acc_str += '{:.2f} '.format(acc)
    logger.info(acc_str)
