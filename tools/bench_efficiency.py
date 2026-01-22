import argparse
import json
import time
import subprocess

import torch

from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier


def _get_gpu_util():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        return float(util)
    except Exception:
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL)
            line = out.decode('utf-8').strip().splitlines()[0]
            return float(line)
        except Exception:
    return None


def _get_ram_mb():
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    process = psutil.Process()
    return float(process.memory_info().rss) / (1024 ** 2)


def _get_cpu_util():
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return float(psutil.cpu_percent(interval=None))


def _forward_logits(model, img):
    feats = model.extract_feat(img)
    return model.head.simple_test(feats, softmax=False, post_process=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark inference efficiency')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-batches', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
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

    total_samples = 0
    total_time = 0.0
    cpu_utils = []
    gpu_utils = []

    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    data_iter = iter(test_loader)
    for _ in range(args.warmup):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(test_loader)
            data = next(data_iter)
        with torch.no_grad():
            _ = _forward_logits(model, data['img'].to(args.device))

    for _ in range(args.num_batches):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(test_loader)
            data = next(data_iter)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = _forward_logits(model, data['img'].to(args.device))
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_samples += data['img'].size(0)
        cpu_util = _get_cpu_util()
        if cpu_util is not None:
            cpu_utils.append(cpu_util)
        util = _get_gpu_util()
        if util is not None:
            gpu_utils.append(util)

    if args.device.startswith('cuda'):
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        vram_mb = 0.0
    ram_mb = _get_ram_mb()

    latency_ms = (total_time / max(total_samples, 1)) * 1000.0
    fps = float(total_samples) / max(total_time, 1e-12)
    results = dict(
        latency_ms=latency_ms,
        fps=fps,
        vram_mb=vram_mb,
        ram_mb=ram_mb,
        cpu_util_avg=float(sum(cpu_utils) / len(cpu_utils)) if cpu_utils else None,
        gpu_util_avg=float(sum(gpu_utils) / len(gpu_utils)) if gpu_utils else None,
    )

    output = json.dumps(results, indent=2, ensure_ascii=True)
    print(output)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write(output + '\n')


if __name__ == '__main__':
    main()
