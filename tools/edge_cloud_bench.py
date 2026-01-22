import argparse
import json
import time

import torch

from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier

from mmfscil.edge_cloud.server import EdgeCloudServer


def _forward_logits(model, img):
    feats = model.extract_feat(img)
    return model.head.simple_test(feats, softmax=False, post_process=False)


def _evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            img = data['img'].to(device)
            gt = data['gt_label'].to(device)
            logits = _forward_logits(model, img)
            pred = torch.argmax(logits, dim=1)
            correct += torch.sum(pred == gt).item()
            total += gt.numel()
    return float(correct) / max(total, 1) * 100.0


def _count_state_bytes(state_dict):
    total = 0
    for value in state_dict.values():
        total += value.numel() * value.element_size()
    return total


def parse_args():
    parser = argparse.ArgumentParser(description='Edge-cloud training benchmark')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-clients', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--partition', choices=['iid', 'by_class'], default='iid')
    parser.add_argument('--classes-per-client', type=int, default=None)
    parser.add_argument('--reflect', action='store_true')
    parser.add_argument('--mutual-eval', action='store_true')
    parser.add_argument('--eval-batches', type=int, default=10)
    parser.add_argument('--out', default=None, help='path to json output')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained = None
    model = build_classifier(cfg.model)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device)

    train_dataset = build_dataset(cfg.data.train if isinstance(cfg.data.train, dict) else cfg.data.train.dataset)
    val_dataset = build_dataset(cfg.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        round_up=False,
    )

    edge_cfg = cfg.get('edge_cloud', {})
    edge_cfg['num_clients'] = int(args.num_clients)
    edge_cfg['local_epochs'] = int(args.local_epochs)
    edge_cfg['partition'] = dict(type=args.partition, classes_per_client=args.classes_per_client, seed=0)
    edge_cfg.setdefault('mutual_eval', {})['enabled'] = bool(args.mutual_eval)
    edge_cfg.setdefault('reflect', {})['enabled'] = bool(args.reflect)
    cfg.edge_cloud = edge_cfg

    server = EdgeCloudServer(model, cfg, args.device)
    clients = server.build_clients(train_dataset)

    share_backbone = bool(edge_cfg.get('share_backbone', True))
    share_neck = bool(edge_cfg.get('share_neck', True))
    share_head = bool(edge_cfg.get('share_head', True))
    train_backbone = bool(edge_cfg.get('train_backbone', True))
    optimizer_cfg = edge_cfg.get('local_optimizer', cfg.optimizer)

    state_bytes = _count_state_bytes(server._select_state_dict(share_backbone, share_neck, share_head))

    acc_curve = []
    start_time = time.perf_counter()
    for _ in range(args.rounds):
        server.run_rounds(
            clients,
            input_mode='images',
            global_rounds=1,
            local_epochs=args.local_epochs,
            optimizer_cfg=optimizer_cfg,
            share_backbone=share_backbone,
            share_neck=share_neck,
            share_head=share_head,
            train_backbone=train_backbone,
            log_prefix='bench',
        )
        acc = _evaluate(server.model, val_loader, args.device, max_batches=args.eval_batches)
        acc_curve.append(acc)
    elapsed = time.perf_counter() - start_time

    total_bytes = state_bytes * args.num_clients * args.rounds * 2
    stability = float(torch.tensor(acc_curve).std().item()) if acc_curve else 0.0

    results = dict(
        num_clients=args.num_clients,
        partition=args.partition,
        rounds=args.rounds,
        acc_curve=acc_curve,
        acc_final=acc_curve[-1] if acc_curve else 0.0,
        total_bytes=total_bytes,
        stability_std=stability,
        elapsed_sec=elapsed,
    )
    output = json.dumps(results, indent=2, ensure_ascii=True)
    print(output)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write(output + '\n')


if __name__ == '__main__':
    main()
