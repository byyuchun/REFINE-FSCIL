_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(type='MLPFFNNeck', in_channels=640, out_channels=512),
    head=dict(
        type='LinearClsHeadCIL',
        num_classes=100,
        eval_classes=60,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True,
        metric_type='linear',
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixupTwoLabel', alpha=0.8, num_classes=-1, prob=0.4),
        dict(type='BatchCutMixTwoLabel', alpha=1.0, num_classes=-1, prob=0.4),
        dict(type='IdentityTwoLabel', num_classes=-1, prob=0.2),
    ]),
)
