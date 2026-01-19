import argparse

import torch
from mmcv import Config

from mmcls.models import build_classifier


def _to_int(val):
    if isinstance(val, (list, tuple)):
        return int(val[0])
    return int(val)


def _infer_img_size(cfg):
    if hasattr(cfg, 'img_size'):
        return _to_int(cfg.img_size)
    data_cfg = cfg.data.get('train')
    if data_cfg is None:
        return 32
    if isinstance(data_cfg, dict) and data_cfg.get('type') == 'RepeatDataset':
        data_cfg = data_cfg.get('dataset', {})
    pipeline = data_cfg.get('pipeline', [])
    for step in pipeline:
        if step.get('type') == 'RandomResizedCrop' and 'size' in step:
            return _to_int(step['size'])
        if step.get('type') == 'CenterCrop' and 'crop_size' in step:
            return _to_int(step['crop_size'])
    return 32


def main():
    parser = argparse.ArgumentParser(description='Prototype sanity check')
    parser.add_argument('config', help='config path')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    img_size = args.img_size or _infer_img_size(cfg)

    model = build_classifier(cfg.model)
    model.init_weights()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()

    num_classes = cfg.model.head.get('num_classes', 100)
    imgs = torch.randn(args.batch_size, 3, img_size, img_size, device=device)
    labels = torch.randint(0, num_classes, (args.batch_size,), device=device)

    losses = model(return_loss=True, img=imgs, gt_label=labels)
    loss = losses['loss']
    loss.backward()
    print('sanity ok: loss={:.6f}'.format(loss.item()))


if __name__ == '__main__':
    main()
