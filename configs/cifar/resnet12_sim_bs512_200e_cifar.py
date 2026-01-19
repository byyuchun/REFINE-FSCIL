_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py',
    '../_base_/default_runtime.py'
]

PROTO_MODE = 'SIM'
SIM_CFG = dict(
    # SIM: pre-assign fixed targets via semantic similarity S to mitigate
    # target conflict and feature-classifier misalignment across sessions.
    sim_path=None,
    sim_format=None,
    sim_eps=1e-6,
    eig_tol=1e-6,
    hungarian=False,
    hungarian_max_classes=200,
    fallback_steps=100,
    fallback_lr=0.1,
)

# model settings
model = dict(
    neck=dict(type='MLPFFNNeck', in_channels=640, out_channels=512),
    head=dict(
        type='ETFHead',
        in_channels=512,
        with_len=False,
        proto_mode=PROTO_MODE,
        sim_cfg=SIM_CFG,
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixupTwoLabel', alpha=0.8, num_classes=-1, prob=0.4),
        dict(type='BatchCutMixTwoLabel', alpha=1.0, num_classes=-1, prob=0.4),
        dict(type='IdentityTwoLabel', num_classes=-1, prob=0.2),
    ]),
)
