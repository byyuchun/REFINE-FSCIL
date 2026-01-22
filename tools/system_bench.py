import argparse
import json
import time

import torch

from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier


def _simulate_network_time(num_bytes, bandwidth_mbps, rtt_ms, loss_rate):
    if bandwidth_mbps <= 0:
        return 0.0
    transfer = (num_bytes * 8.0) / (bandwidth_mbps * 1e6)
    if loss_rate > 0:
        transfer = transfer / max(1e-6, (1.0 - loss_rate))
    return transfer + (rtt_ms / 1000.0)


def _forward_logits(model, img):
    feats = model.extract_feat(img)
    return model.head.simple_test(feats, softmax=False, post_process=False)


def parse_args():
    parser = argparse.ArgumentParser(description='System benchmark for edge/cloud modes')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--mode', choices=['cloud-only', 'edge-only', 'edge-cloud'], default='edge-cloud')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-batches', type=int, default=50)
    parser.add_argument('--bandwidth-mbps', type=float, default=10.0)
    parser.add_argument('--rtt-ms', type=float, default=20.0)
    parser.add_argument('--loss-rate', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=None)
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
    model.eval()

    test_dataset_cfg = cfg.data.test
    test_dataset = build_dataset(test_dataset_cfg)
    loader_cfg = dict(
        num_gpus=1,
        dist=False,
        round_up=False,
        seed=cfg.get('seed', None),
    )
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}
    if args.batch_size is not None:
        test_loader_cfg['samples_per_gpu'] = args.batch_size
    test_loader = build_dataloader(test_dataset, **test_loader_cfg)

    total_latency = 0.0
    total_bytes = 0
    total_samples = 0
    correct = 0

    data_iter = iter(test_loader)
    for _ in range(args.num_batches):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(test_loader)
            data = next(data_iter)
        img = data['img'].to(args.device)
        gt = data['gt_label'].to(args.device)
        batch_bytes = img.numel() * img.element_size()

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        if args.mode == 'cloud-only':
            net_time = _simulate_network_time(batch_bytes, args.bandwidth_mbps, args.rtt_ms, args.loss_rate)
            with torch.no_grad():
                logits = _forward_logits(model, img)
        elif args.mode == 'edge-only':
            net_time = 0.0
            with torch.no_grad():
                logits = _forward_logits(model, img)
        else:
            with torch.no_grad():
                logits = _forward_logits(model, img)
            logit_bytes = logits.numel() * logits.element_size()
            net_time = _simulate_network_time(logit_bytes, args.bandwidth_mbps, args.rtt_ms, args.loss_rate)
            batch_bytes = logit_bytes
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - start
        total_latency += compute_time + net_time
        total_bytes += int(batch_bytes)
        total_samples += img.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += torch.sum(pred == gt).item()

    avg_latency_ms = (total_latency / max(total_samples, 1)) * 1000.0
    acc = float(correct) / max(total_samples, 1) * 100.0
    results = dict(
        mode=args.mode,
        avg_latency_ms=avg_latency_ms,
        uplink_mb=total_bytes / (1024 ** 2),
        acc=acc,
        bandwidth_mbps=args.bandwidth_mbps,
        rtt_ms=args.rtt_ms,
        loss_rate=args.loss_rate,
    )
    output = json.dumps(results, indent=2, ensure_ascii=True)
    print(output)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write(output + '\n')


if __name__ == '__main__':
    main()
