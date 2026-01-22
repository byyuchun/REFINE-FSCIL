# model settings
# Fixed prototype mode. ETF enforces maximal angular separation (uniform
# simplex); SIM encodes inter-class relations via a similarity matrix to
# pre-assign and fix optimal alignment targets, avoiding target conflict.
PROTO_MODE = 'ETF'
SIM_CFG = dict(
    # SIM encodes class relations from S (C x C); fallback uses S=I to keep
    # the training runnable without external semantics.
    sim_path=None,
    sim_format=None,
    sim_eps=1e-6,
    eig_tol=1e-6,
    hungarian=False,
    hungarian_max_classes=200,
    align_strategy=None,  # none/hungarian/greedy/random
    align_seed=0,
    fallback_steps=100,
    fallback_lr=0.1,
)
model = dict(
    type='ImageClassifierCIL',
    backbone=dict(
        type='ResNet12',
        with_avgpool=False,
        flatten=False
    ),
    neck=dict(type='MLPNeck', in_channels=640, out_channels=512),
    head=dict(
        type='ETFHead',
        num_classes=100,
        eval_classes=60,
        in_channels=512,
        loss=dict(type='DRLoss', loss_weight=10.),
        topk=(1, 5),
        cal_acc=True,
        proto_mode=PROTO_MODE,
        sim_cfg=SIM_CFG,
    )
)
